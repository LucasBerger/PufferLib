
#include "quarryboard.h"
#include <string.h>

#define MAX_TEST_BOARD 80

// Helper math
static unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int rand_range(unsigned int* rng, int min, int max) {
    if (min >= max) return min;
    return min + (xorshift32(rng) % (max - min + 1));
}

// ----------------------------------------------------------------------------
// Shape & Deck Logic (Mostly copied from crystalboard.c)
// ----------------------------------------------------------------------------
static const int SHAPE_L[4][2] = {{0,0}, {0,1}, {0,2}, {1,2}};
static const int SHAPE_T[4][2] = {{1,0}, {0,1}, {1,1}, {1,2}};
static const int SHAPE_LITTLE_L[3][2] = {{0,0}, {1,0}, {0,1}};
static const int SHAPE_Z[4][2] = {{1,0}, {2,0}, {0,1}, {1,1}};
static const int SHAPE_I[4][2] = {{0,0}, {1,0}, {2,0}, {3,0}};
static const int SHAPE_LITTLE_I[3][2] = {{0,0}, {1,0}, {2,0}};
static const int SHAPE_BOX[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};
static const int SHAPE_BASE[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};
static const int SHAPE_BASE_SIZE = 4;

static const int* ALL_SHAPES[] = {
    (const int*)SHAPE_L, (const int*)SHAPE_T, (const int*)SHAPE_LITTLE_L,
    (const int*)SHAPE_Z, (const int*)SHAPE_I, (const int*)SHAPE_LITTLE_I, (const int*)SHAPE_BOX
};
static const int ALL_SHAPE_SIZES[] = {4, 4, 3, 4, 4, 3, 4};
#define NUM_SHAPE_TYPES 7

static void copy_shape(Card* card, int shape_idx) {
    const int* src = ALL_SHAPES[shape_idx];
    int size = ALL_SHAPE_SIZES[shape_idx];
    card->shape.num_cells = size;
    for (int i = 0; i < size; i++) {
        card->shape.cells[i][0] = src[i*2];
        card->shape.cells[i][1] = src[i*2 + 1];
    }
}

static void init_card_decks(Quarryboard* env) {
    env->building_deck_size = 21;
    int deck_idx = 0;
    // 21 cards: 7 shapes * 3 copies.
    // We want some to be quarries.
    // Let's say: For each shape, 2 extraction, 1 quarry.
    for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
        env->building_deck[deck_idx++] = shape; // Extraction
        env->building_deck[deck_idx++] = shape; // Extraction
        env->building_deck[deck_idx++] = shape | 0x100; // Quarry (bit 8 set)
    }

    env->street_deck_size = 21;
    deck_idx = 0;
    for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
        for (int copy = 0; copy < 3; copy++) env->street_deck[deck_idx++] = shape;
    }
}

static void shuffle_deck(int* deck, int size, unsigned int* rng_state) {
    for (int i = size - 1; i > 0; i--) {
        int j = xorshift32(rng_state) % (i + 1);
        int tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }
}

static void draw_random_card(Card* card, int* deck, int* deck_size, int is_building, unsigned int* rng_state) {
    if (*deck_size == 0) {
        *deck_size = 21;
        // Reshuffle with original distribution
        int deck_idx = 0;
        if (is_building) {
            for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
                deck[deck_idx++] = shape;
                deck[deck_idx++] = shape;
                deck[deck_idx++] = shape | 0x100;
            }
        } else {
            for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
                for (int copy = 0; copy < 3; copy++) deck[deck_idx++] = shape;
            }
        }
        shuffle_deck(deck, *deck_size, rng_state);
    }
    int deck_idx = xorshift32(rng_state) % *deck_size;
    int val = deck[deck_idx];
    int shape_idx = val & 0xFF;
    int is_quarry = (val >> 8) & 1;
    
    deck[deck_idx] = deck[*deck_size - 1];
    (*deck_size)--;

    copy_shape(card, shape_idx);
    card->is_building = is_building;
    card->is_quarry = is_quarry;
    card->static_points = is_building ? ((card->shape.num_cells == 3) ? 3 : 2) : 1;
}

static void replace_used_card(Quarryboard* env, int card_idx) {
    if (card_idx < NUM_BUILDING_SLOTS) {
        draw_random_card(&env->building_market[card_idx], env->building_deck,
                        &env->building_deck_size, 1, &env->rng_state);
    } else {
        int street_idx = card_idx - NUM_BUILDING_SLOTS;
        draw_random_card(&env->street_market[street_idx], env->street_deck,
                        &env->street_deck_size, 0, &env->rng_state);
    }
}

static void init_decks(Quarryboard* env) {
    init_card_decks(env);
    shuffle_deck(env->building_deck, env->building_deck_size, &env->rng_state);
    shuffle_deck(env->street_deck, env->street_deck_size, &env->rng_state);
    
    // Draw initial building market with guaranteed variety
    // Ensure at least 1-2 quarries in first 4 slots
    int quarry_count = 0;
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        draw_random_card(&env->building_market[i], env->building_deck,
                        &env->building_deck_size, 1, &env->rng_state);
        if (env->building_market[i].is_quarry) quarry_count++;
    }
    
    // If no quarries drawn, force replace one card with a quarry
    if (quarry_count == 0) {
        // Find a quarry in remaining deck or reshuffle
        int found_quarry = 0;
        for (int i = 0; i < env->building_deck_size; i++) {
            if ((env->building_deck[i] >> 8) & 1) {
                // Found quarry - replace slot 0
                int val = env->building_deck[i];
                int shape_idx = val & 0xFF;
                
                // Remove from deck
                env->building_deck[i] = env->building_deck[env->building_deck_size - 1];
                env->building_deck_size--;
                
                // Replace market slot
                copy_shape(&env->building_market[0], shape_idx);
                env->building_market[0].is_building = 1;
                env->building_market[0].is_quarry = 1;
                env->building_market[0].static_points = (env->building_market[0].shape.num_cells == 3) ? 3 : 2;
                
                found_quarry = 1;
                break;
            }
        }
    }
    
    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        draw_random_card(&env->street_market[i], env->street_deck,
                        &env->street_deck_size, 0, &env->rng_state);
    }
}

// ----------------------------------------------------------------------------
// Board / Map Logic
// ----------------------------------------------------------------------------
static void place_base(Quarryboard* env, int player_id, int x, int y) {
    // Check bounds for 2x2 base
    for (int i = 0; i < SHAPE_BASE_SIZE; i++) {
        int bx = x + SHAPE_BASE[i][0];
        int by = y + SHAPE_BASE[i][1];
        if (bx >= 0 && bx < env->width && by >= 0 && by < env->height) {
            int idx = by * env->width + bx;
            env->board_owner[idx] = player_id + 1; // 1-based owner
            env->board_structure[idx] = TILE_START_BASE;
        }
    }
}

static void place_crystal(Quarryboard* env, int index, int x, int y) {
    if (x < 0) x = 0; if (x >= env->width) x = env->width - 1;
    if (y < 0) y = 0; if (y >= env->height) y = env->height - 1;
    
    env->crystals[index].x = x;
    env->crystals[index].y = y;
    env->crystals[index].resources = MAX_CRYSTAL_RESOURCES;
    for(int i=0; i<NUM_PLAYERS; i++) env->crystals[index].connections[i] = 0;

    int idx = y * env->width + x;
    if (idx < env->board_tiles) {
        env->board_feature[idx] = TILE_CRYSTAL_NODE;
    }
}

static void setup_map(Quarryboard* env) {
    int margin = 4;
    int w = env->width;
    int h = env->height;
    
    // Player Bases
    // P1: Top-Left (margin, margin)
    // P2: Top-Right (w-margin-2, margin) - minus 2 for 2x2 shape
    // P3: Bot-Right (w-margin-2, h-margin-2)
    // P4: Bot-Left (margin, h-margin-2)
    
    // Actually user said "placed in the corner with 4 margin to the edges"
    // Let's assume margin is the coordinate of the top-left of the base?
    
    int p1_x = margin, p1_y = margin;
    int p2_x = w - margin - 2, p2_y = margin;
    int p3_x = w - margin - 2, p3_y = h - margin - 2;
    int p4_x = margin, p4_y = h - margin - 2;
    
    place_base(env, 0, p1_x, p1_y);
    place_base(env, 1, p2_x, p2_y);
    place_base(env, 2, p3_x, p3_y);
    place_base(env, 3, p4_x, p4_y);
    
    int c_idx = 0;
    
    // 1. Starter Crystals (Manhattan 3 or 4 towards center/opponents)
    // P1: Direction (+1, +1)
    {
        int dist = 3 + (xorshift32(&env->rng_state) % 2);
        // Randomly split dist into dx, dy (avoid 0 if possible to be diagonal-ish?)
        // "direction of the center". (+, +).
        // Let's just pick x+d, y or x, y+d or mix.
        // Simple: dx = rand(0, dist), dy = dist - dx.
        int dx = rand_range(&env->rng_state, 0, dist);
        int dy = dist - dx;
        // Base center is approx px+0.5, py+0.5. Crystal is adjacent.
        // Let's place relative to base corner (p1_x+1, p1_y+1) is 'center' of base.
        place_crystal(env, c_idx++, p1_x + 1 + dx, p1_y + 1 + dy);
    }
    // P2: Direction (-1, +1)
    {
        int dist = 3 + (xorshift32(&env->rng_state) % 2);
        int dx = rand_range(&env->rng_state, 0, dist);
        int dy = dist - dx;
        place_crystal(env, c_idx++, p2_x - dx, p2_y + 1 + dy);
    }
    // P3: Direction (-1, -1)
    {
        int dist = 3 + (xorshift32(&env->rng_state) % 2);
        int dx = rand_range(&env->rng_state, 0, dist);
        int dy = dist - dx;
        place_crystal(env, c_idx++, p3_x - dx, p3_y - dy);
    }
    // P4: Direction (+1, -1)
    {
        int dist = 3 + (xorshift32(&env->rng_state) % 2);
        int dx = rand_range(&env->rng_state, 0, dist);
        int dy = dist - dx;
        place_crystal(env, c_idx++, p4_x + 1 + dx, p4_y - dy);
    }
    
    // 2. Side Crystals (2-line band)
    // Center Bands: Vertical x=[w/2-1, w/2], Horizontal y=[h/2-1, h/2]
    int mid_x_start = w/2 - 1;
    int mid_y_start = h/2 - 1;
    
    // Top Side (P1-P2): Vertical band? User: "inbetween each player... on the 2 line band in the center".
    // Between P1 and P2 is the Vertical Center Band (x axis separation).
    // range y: [margin, h/2-1] roughly?
    // "random inside their respective bounds".
    // Top Crystal: x in [mid_x_start, mid_x_start+1], y in [margin, h/2 - 2]
    place_crystal(env, c_idx++, 
        rand_range(&env->rng_state, mid_x_start, mid_x_start+1),
        rand_range(&env->rng_state, margin, h/2 - 2)
    );
    
    // Right Crystal (P2-P3): Horizontal band (y mid), x right side.
    place_crystal(env, c_idx++,
        rand_range(&env->rng_state, w/2 + 2, w - margin), // Right side
        rand_range(&env->rng_state, mid_y_start, mid_y_start+1)
    );
    
    // Bottom Crystal (P3-P4): Vertical band, y bottom.
    place_crystal(env, c_idx++,
        rand_range(&env->rng_state, mid_x_start, mid_x_start+1),
        rand_range(&env->rng_state, h/2 + 2, h - margin)
    );
    
    // Left Crystal (P4-P1): Horizontal band, x left.
    place_crystal(env, c_idx++,
        rand_range(&env->rng_state, margin, w/2 - 2),
        rand_range(&env->rng_state, mid_y_start, mid_y_start+1)
    );
    
    // 3. Center Crystals (2 in 6x6 box)
    // Box center: w/2, h/2. Offset -3 to +2.
    for (int i=0; i<2; i++) {
        place_crystal(env, c_idx++,
            rand_range(&env->rng_state, w/2 - 3, w/2 + 2),
            rand_range(&env->rng_state, h/2 - 3, h/2 + 2)
        );
    }
}

// ----------------------------------------------------------------------------
// Game Logic
// ----------------------------------------------------------------------------

static void rotate_shape(const int shape[][2], int num_cells, int rotation, int out_shape[][2]) {
    int temp[MAX_SHAPE_SIZE][2];
    for (int i = 0; i < num_cells; i++) {
        temp[i][0] = shape[i][0];
        temp[i][1] = shape[i][1];
    }
    for (int r = 0; r < (rotation % 4); r++) {
        for (int i = 0; i < num_cells; i++) {
            int x = temp[i][0], y = temp[i][1];
            temp[i][0] = y;
            temp[i][1] = -x;
        }
    }
    int min_x = temp[0][0], min_y = temp[0][1];
    for (int i = 1; i < num_cells; i++) {
        if (temp[i][0] < min_x) min_x = temp[i][0];
        if (temp[i][1] < min_y) min_y = temp[i][1];
    }
    for (int i = 0; i < num_cells; i++) {
        out_shape[i][0] = temp[i][0] - min_x;
        out_shape[i][1] = temp[i][1] - min_y;
    }
}

static int can_place_tile(Quarryboard* env, int x, int y, int shape[][2], int num_cells) {
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        if (bx < 0 || bx >= env->width || by < 0 || by >= env->height) return 0;
        int idx = by * env->width + bx;
        if (env->board_owner[idx] != 0 || env->board_feature[idx] != TILE_EMPTY) return 0;
    }
    return 1;
}

static int is_connected_to_network(Quarryboard* env, int player_id, int x, int y, int shape[][2], int num_cells) {
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    int owner_val = player_id + 1;
    
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        for (int d = 0; d < 4; d++) {
            int nx = bx + dx[d];
            int ny = by + dy[d];
            if (nx < 0 || nx >= env->width || ny < 0 || ny >= env->height) continue;
            int idx = ny * env->width + nx;
            if (env->board_owner[idx] == owner_val) {
                // FIXED: Only streets and bases count as connection points
                // Extractors and quarries do NOT allow connection
                if (env->board_structure[idx] == TILE_STREET || 
                    env->board_structure[idx] == TILE_START_BASE) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

// Helper for flood fill to mark connected building components
static void mark_component(Quarryboard* env, unsigned char* visited, int start_x, int start_y, int owner) {
    // Simple DFS or BFS. Using explicit stack for 30x30 board.
    int stack[MAX_TEST_BOARD * MAX_TEST_BOARD]; // Max size
    int top = 0;
    
    stack[top++] = start_y * env->width + start_x;
    visited[start_y * env->width + start_x] = 1;
    
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    while (top > 0) {
        int curr = stack[--top];
        int cx = curr % env->width;
        int cy = curr / env->width;
        
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            
            if (nx >= 0 && nx < env->width && ny >= 0 && ny < env->height) {
                int n_idx = ny * env->width + nx;
                if (!visited[n_idx] && 
                    env->board_owner[n_idx] == owner && 
                    env->board_structure[n_idx] == TILE_EXTRACTION_BUILDING) {
                    
                    visited[n_idx] = 1;
                    stack[top++] = n_idx;
                }
            }
        }
    }
}

static void update_crystal_connections(Quarryboard* env) {
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    // Visited array to track unique building components per crystal check
    // Allocated once here? Or on stack? 30x30 is small.
    // Dynamic size based on env->board_tiles
    unsigned char* visited = (unsigned char*)calloc(env->board_tiles, 1);
    
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        if (env->crystals[i].resources <= 0) {
            // Depleted
            for(int p=0; p<NUM_PLAYERS; p++) env->crystals[i].connections[p] = 0;
            continue;
        }
        
        // Reset connections
        for(int p=0; p<NUM_PLAYERS; p++) env->crystals[i].connections[p] = 0;
        
        // Reset visited for this crystal's context
        // We only really need to track visited for the neighbors/components of THIS crystal.
        memset(visited, 0, env->board_tiles);
        
        int cx = env->crystals[i].x;
        int cy = env->crystals[i].y;
        
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx < 0 || nx >= env->width || ny < 0 || ny >= env->height) continue;
            
            int idx = ny * env->width + nx;
            int owner = env->board_owner[idx];
            
            // If it's an extraction building and we haven't visited this component yet
            if (owner > 0 && 
                env->board_structure[idx] == TILE_EXTRACTION_BUILDING && 
                !visited[idx]) {
                
                // Found a new unique building connected to this crystal
                env->crystals[i].connections[owner-1]++;
                
                // Mark the entire component as visited so other parts of it 
                // touching the crystal don't count again.
                mark_component(env, visited, nx, ny, owner);
            }
        }
    }
    
    free(visited);
}

static float pending_rewards[NUM_PLAYERS];

static void distribute_resources(Quarryboard* env) {
    // 1. Quarries generate stones
    // Each quarry building generates 1 stone per round
    // This function is called when active_turns % NUM_PLAYERS == 0
    for(int p = 0; p < NUM_PLAYERS; p++) {
        for(int i = 0; i < env->quarry_count[p]; i++) {
            env->stones[p]++;
            
            // Track stats
            if (p==0) env->acc_p1_stone_collected += 1.0f;
            else if (p==1) env->acc_p2_stone_collected += 1.0f;
            else if (p==2) env->acc_p3_stone_collected += 1.0f;
            else if (p==3) env->acc_p4_stone_collected += 1.0f;
        }
    }

    // 2. Crystal Extraction
    // Run for each crystal
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        CrystalNode* c = &env->crystals[i];
        if (c->resources <= 0) continue;
        
        int total_demand = 0; // Total count of connected buildings across all players
        int demands[NUM_PLAYERS];
        int active_players = 0;
        
        for (int p=0; p<NUM_PLAYERS; p++) {
            demands[p] = c->connections[p];
            total_demand += demands[p]; // 1 demand per building
            if (demands[p] > 0) active_players++;
        }
        
        if (total_demand == 0) continue;
        
        if (c->resources >= total_demand) {
            // Enough for everyone
            for (int p=0; p<NUM_PLAYERS; p++) {
                if (demands[p] > 0) {
                    env->scores[p] += demands[p];
                    env->cumulative_crystals[p] += demands[p];
                    pending_rewards[p] += (float)demands[p] * REWARD_RESOURCE;
                    // Track log
                    float* collected_log = &env->acc_p1_crystal_collected;
                    collected_log[p] += (float)demands[p];
                }
            }
            c->resources -= total_demand;
        } else {
            // Scarcity logic
             while (c->resources > 0 && active_players > 0) {
                 if (c->resources >= active_players) {
                     for (int p=0; p<NUM_PLAYERS; p++) {
                         if (demands[p] > 0) {
                             env->scores[p] += 1;
                             env->cumulative_crystals[p] += 1;
                             pending_rewards[p] += REWARD_RESOURCE;
                             // Track log
                             float* collected_log = &env->acc_p1_crystal_collected;
                             collected_log[p] += 1.0f;
                         }
                     }
                     c->resources -= active_players;
                 } else {
                     // Ultra scarcity: Not enough for 1 per player.
                     // Find max connections
                     int max_con = -1;
                     int count_max = 0;
                     int best_p = -1;
                     
                     for(int p=0; p<NUM_PLAYERS; p++) {
                         if (demands[p] > max_con) {
                             max_con = demands[p];
                             count_max = 1;
                             best_p = p;
                         } else if (demands[p] == max_con && demands[p] > 0) {
                             count_max++;
                         }
                     }
                     
                     if (count_max == 1) {
                         // Winner
                         env->scores[best_p] += 1;
                         env->cumulative_crystals[best_p] += 1;
                         pending_rewards[best_p] += REWARD_RESOURCE;
                         // Track log
                         float* collected_log = &env->acc_p1_crystal_collected;
                         collected_log[best_p] += 1.0f;
                         c->resources -= 1;
                     } else {
                         // Tie -> Deplete all remaining
                         c->resources = 0;
                     }
                 }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Step & Observation
// ----------------------------------------------------------------------------

static int encode_action_comp(Quarryboard* env, int card_idx, int x, int y, int rotation) {
    int pos_idx = y * env->width + x;
    return card_idx * (env->width * env->height * 4) + pos_idx * 4 + rotation;
}

static void decode_action_comp(Quarryboard* env, int action, int* card_idx, int* x, int* y, int* rotation) {
    *rotation = action % 4;
    int rem = action / 4;
    int pos_idx = rem % (env->width * env->height);
    *x = pos_idx % env->width;
    *y = pos_idx / env->width;
    *card_idx = rem / (env->width * env->height);
}

// Relative Observation Logic
static float get_obs_owner(int actual_owner, int current_player) {
    if (actual_owner == 0) return 0.0f;
    // 1-based owner.
    // 1=P1, 2=P2 ...
    // Rel 1 = Me.
    // Rel 2 = Next.
    int p_idx = actual_owner - 1; // 0..3
    int rel = ((p_idx - current_player + NUM_PLAYERS) % NUM_PLAYERS) + 1;
    
    // Scheme:
    // Structure Type: 1..4 (Street, Bldg, Base, Crystal-Feature(4))
    // Owner:
    //   Me (10)
    //   Opp1 (20)
    //   Opp2 (30)
    //   Opp3 (40)
    
    int offset = rel * 10;
    return (float)offset;
}

static void fill_observation_comp(Quarryboard* env) {
    int p = env->current_player;
    int idx = 0;
    
    // Board
    for (int i = 0; i < env->board_tiles; i++) {
        float val = 0.0f;
        int type = env->board_structure[i];
        if (env->board_feature[i] == TILE_CRYSTAL_NODE) type = 4; // Feature overrides structure visual?
        val = (float)type;
        
        int owner = env->board_owner[i];
        if (owner != 0) {
            val += get_obs_owner(owner, p);
        }
        
        env->observations[idx++] = val / 44.0f; // Normalized
    }
    
    // Market
    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) ? &env->building_market[c] : &env->street_market[c - NUM_BUILDING_SLOTS];
        env->observations[idx++] = card->is_building ? 1.0f : 0.0f;
        env->observations[idx++] = card->static_points / 5.0f;
        env->observations[idx++] = card->shape.num_cells / 5.0f;
        
        float shape_grid[25] = {0};
        int raw_shape[MAX_SHAPE_SIZE][2];
        for(int k=0; k<card->shape.num_cells; k++) {
           raw_shape[k][0] = card->shape.cells[k][0];
           raw_shape[k][1] = card->shape.cells[k][1];
           // Do we rotate market shape for obs?
           // Original didn't.
        }
        
        for (int k = 0; k < card->shape.num_cells; k++) {
            int sx = raw_shape[k][0];
            int sy = raw_shape[k][1];
            if (sx >= 0 && sx < 5 && sy >= 0 && sy < 5) shape_grid[sy * 5 + sx] = 1.0f;
        }
        for (int k = 0; k < 25; k++) env->observations[idx++] = shape_grid[k];
        
        // Resource Costs
        int cost_c = 0;
        int cost_s = 0;
        if (card->is_building) {
            if (card->is_quarry) { cost_c = 1; cost_s = 2; }
            else { cost_c = 2; cost_s = 1; }
        } else {
            // Street: 1 Crystal + 1 Stone
            cost_c = 1;
            cost_s = 1;
        }
        
        env->observations[idx++] = (float)cost_c / 5.0f;
        env->observations[idx++] = (float)cost_s / 5.0f;
    }
    
    // Crystal Info (Relative connections)
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        env->observations[idx++] = env->crystals[i].x / (float)(env->width - 1);
        env->observations[idx++] = env->crystals[i].y / (float)(env->height - 1);
        
        // Connections: My, Opp1, Opp2, Opp3
        for (int k=0; k<NUM_PLAYERS; k++) {
            int target_p = (p + k) % NUM_PLAYERS;
            float conn = (float)env->crystals[i].connections[target_p];
            env->observations[idx++] = conn / 5.0f; // Max 5 connections? Just scaling.
        }
        
        env->observations[idx++] = env->crystals[i].resources / 10.0f;
    }
    
    // Player Resources (Relative: Me, Next, Next+1, Next+2)
    for(int k=0; k<NUM_PLAYERS; k++) {
        int target_p = (p + k) % NUM_PLAYERS;
        env->observations[idx++] = (float)env->scores[target_p] / 20.0f;
        env->observations[idx++] = (float)env->stones[target_p] / 20.0f;
        env->observations[idx++] = (float)env->cumulative_crystals[target_p] / 50.0f;
        env->observations[idx++] = (float)env->finished[target_p];
        env->observations[idx++] = (float)env->dead[target_p];
    }

    // Sync Info (3 floats)
    // 0: Current Player (0-3)
    // 1: Human Action Ready (0/1)
    // 2: Last Human Action (int)
    env->observations[idx++] = (float)env->current_player;
    env->observations[idx++] = (float)env->human_action_ready;
    env->observations[idx++] = (float)env->last_human_action;

    // Action Mask
    // Valid for Current Player
    int rotated_shapes[TOTAL_MARKET_SLOTS][4][MAX_SHAPE_SIZE][2];
    int num_cells[TOTAL_MARKET_SLOTS];

    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) 
            ? &env->building_market[c] 
            : &env->street_market[c - NUM_BUILDING_SLOTS];
        
        num_cells[c] = card->shape.num_cells;
        int raw_shape[MAX_SHAPE_SIZE][2];
        for (int k = 0; k < card->shape.num_cells; k++) {
            raw_shape[k][0] = card->shape.cells[k][0];
            raw_shape[k][1] = card->shape.cells[k][1];
        }
        
        for (int r = 0; r < 4; r++) {
            rotate_shape(raw_shape, card->shape.num_cells, r, rotated_shapes[c][r]);
        }
    }
    
    int num_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
    for (int i = 0; i < num_actions; i++) env->observations[idx + i] = 0.0f;
    
    int valid_count = 0;
    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) 
            ? &env->building_market[c] 
            : &env->street_market[c - NUM_BUILDING_SLOTS];

        int affordable = 0;
        if (card->is_building) {
             if (card->is_quarry) affordable = (env->stones[p]>=2 && env->scores[p]>=1);
             else affordable = (env->scores[p]>=2 && env->stones[p]>=1);
        } else {
             // Street: 1 Crystal + 1 Stone
             affordable = (env->scores[p] >= 1 && env->stones[p] >= 1);
        }

        for (int x = 0; x < env->width; x++) {
            for (int y = 0; y < env->height; y++) {
                for (int r = 0; r < 4; r++) {
                    if (affordable && 
                        can_place_tile(env, x, y, rotated_shapes[c][r], num_cells[c]) &&
                        is_connected_to_network(env, p, x, y, rotated_shapes[c][r], num_cells[c])) {
                        
                        int aid = encode_action_comp(env, c, x, y, r);
                        env->observations[idx + aid] = 1.0f;
                        valid_count++;
                    }
                }
            }
        }
    }
    
    // Pass and Finish Masks
    // Pass is always valid (fallback)
    env->observations[idx + num_actions] = 1.0f; 
    
    // Finish is always valid
    env->observations[idx + num_actions + 1] = 1.0f;
    
    env->valid_placements = valid_count; // Stat
}

static int count_connections(Quarryboard* env, int p) {
    int count = 0;
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        int connected = 0;
        int cx = env->crystals[i].x;
        int cy = env->crystals[i].y;
        
        for (int d=0; d<4; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx>=0 && nx < env->width && ny>=0 && ny < env->height) {
                int idx = ny*env->width + nx;
                if (env->board_owner[idx] == (p+1) && 
                    env->board_structure[idx] == TILE_EXTRACTION_BUILDING) {
                    connected = 1;
                    break;
                }
            }
        }
        if (connected) count++;
    }
    return count;
}

static float get_min_dist_player(Quarryboard* env, int p) {
    float total_min_dist = 0;
    
    // For each crystal
    for(int c=0; c<NUM_CRYSTALS; c++) {
        // Check if p connected it by using our helper logic or stored state
        // We do dynamic check to be sure for distance calculation context
        int connected = 0;
        int cx = env->crystals[c].x;
        int cy = env->crystals[c].y;
        int dx[] = {0, 0, 1, -1};
        int dy[] = {1, -1, 0, 0};
        for (int d=0; d<4; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx>=0 && nx < env->width && ny>=0 && ny < env->height) {
                int idx = ny*env->width + nx;
                if (env->board_owner[idx] == (p+1) && 
                    env->board_structure[idx] == TILE_EXTRACTION_BUILDING) {
                    connected = 1;
                    break;
                }
            }
        }
        
        if (connected) continue;
        
        // Find min dist from any of player's tiles
        float min_d = (float)(env->width + env->height);
        
        for(int idx=0; idx<env->board_tiles; idx++) {
            if(env->board_owner[idx] == (p+1)) {
                int px = idx % env->width;
                int py = idx / env->width;
                float d = (float)(abs(px - cx) + abs(py - cy));
                if (d < min_d) min_d = d;
            }
        }
        total_min_dist += min_d;
    }
    return total_min_dist;
}

// ----------------------------------------------------------------------------
// API
// ----------------------------------------------------------------------------

void q_reset(Quarryboard* env) {
    if (!env->board_owner) return;
    
    memset(env->board_owner, 0, env->board_tiles);
    memset(env->board_structure, TILE_EMPTY, env->board_tiles);
    memset(env->board_feature, TILE_EMPTY, env->board_tiles);
    memset(pending_rewards, 0, sizeof(pending_rewards));
    
    env->reset_count++;
    env->rng_state = env->seed + env->reset_count * 1003;
    if (env->rng_state == 0) env->rng_state = 1;
    
    env->tick = 0;
    env->active_turns = 0;
    env->current_player = 0;
    
    // Reset quarry building trackers for all players
    for(int i=0; i<NUM_PLAYERS; i++) {
        env->quarry_count[i] = 0;
    }
    
    // Config defaults (should be set via kwargs/struct? user said make them variables)
    // We'll set defaults here, can be overridden if exposed.
    env->start_crystals = 5;
    env->start_stones = 5;

    for(int i=0; i<NUM_PLAYERS; i++) {
        env->scores[i] = env->start_crystals;
        env->stones[i] = env->start_stones;
        env->cumulative_crystals[i] = 0;
        env->finished[i] = 0;
        env->dead[i] = 0;
    }
    
    env->acc_p1_crystal_collected = 0;
    env->acc_p2_crystal_collected = 0;
    env->acc_p3_crystal_collected = 0;
    env->acc_p4_crystal_collected = 0;
    
    env->acc_p1_crystals_connected_sum = 0;
    env->acc_p2_crystals_connected_sum = 0;
    env->acc_p3_crystals_connected_sum = 0;
    env->acc_p4_crystals_connected_sum = 0;
    
    env->consecutive_passes = 0;
    
    // Do NOT clear env->log here, as it might contain stats from previous episode 
    // waiting to be collected by vec_log.
    
    setup_map(env);
    init_decks(env);
    
    // Init distances
    for(int i=0; i<NUM_PLAYERS; i++) {
        env->last_total_distance[i] = get_min_dist_player(env, i);
    }
    
    env->current_episode_return = 0;
    env->current_p1_return = 0;
    env->current_p2_return = 0;
    env->current_p3_return = 0;
    env->current_p4_return = 0;
    
    // Init Human State
    env->cursor_x = env->width / 2;
    env->cursor_y = env->height / 2;
    env->selected_card = 0;
    env->rotation = 0;
    env->human_action_ready = 0;
    env->last_human_action = 0;
    
    // Clear log state for human action
    env->log.human_action_ready = 0;
    env->log.last_human_action = 0;
    
    fill_observation_comp(env);
}

static int try_spend(Quarryboard* env, int p, int cost_stone, int cost_crystal) {
    if (env->stones[p] >= cost_stone && env->scores[p] >= cost_crystal) {
        env->stones[p] -= cost_stone;
        env->scores[p] -= cost_crystal;
        return 1;
    }
    return 0;
}
static int try_spend_street(Quarryboard* env, int p) {
    if ((env->stones[p] + env->scores[p]) >= 2) {
        int cost = 2;
        int s_paid = (env->stones[p] >= cost) ? cost : env->stones[p];
        env->stones[p] -= s_paid;
        cost -= s_paid;
        if (cost > 0) env->scores[p] -= cost;
        return 1;
    }
    return 0;
}


void q_step(Quarryboard* env) {
    if (env->terminals[0]) {
        q_reset(env);
        env->terminals[0] = 0;
        return;
    }

    int p = env->current_player;
    int action = env->actions[0];
    
    // Check Action Type
    int num_std_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
    int act_pass = num_std_actions;
    int act_finish = num_std_actions + 1;

    // Reset loop rewards for this step
    env->rewards[0] = 0; 
    
    // 1. Process Action
    if (action == act_finish) {
        env->finished[p] = 1;
        // No immediate reward. Proceed to next player.
        
    } else if (action == act_pass) {
        env->rewards[0] += REWARD_PASS_PENALTY;
        env->consecutive_passes++;
        
        // If we pass, do we still consume a turn? Yes.
    } else if (action < num_std_actions) {
        // Decode
        int card_idx, x, y, rotation;
        decode_action_comp(env, action, &card_idx, &x, &y, &rotation);
        
        Card* card;
        if (card_idx < NUM_BUILDING_SLOTS) card = &env->building_market[card_idx];
        else card = &env->street_market[card_idx - NUM_BUILDING_SLOTS];

        // Validity Checks
        int rotated_shape[MAX_SHAPE_SIZE][2];
        int raw_shape[MAX_SHAPE_SIZE][2];
        for (int k=0; k<card->shape.num_cells; k++) {
             raw_shape[k][0] = card->shape.cells[k][0];
             raw_shape[k][1] = card->shape.cells[k][1];
        }
        rotate_shape(raw_shape, card->shape.num_cells, rotation, rotated_shape);
        
        int valid_place = can_place_tile(env, x, y, rotated_shape, card->shape.num_cells);
        int connected = is_connected_to_network(env, p, x, y, rotated_shape, card->shape.num_cells);
        
        // Check affordability FIRST before any other logic
        int affordable = 0;
        if (card->is_building) {
            if (card->is_quarry) {
                // Quarry: 2 Stone, 1 Crystal
                affordable = (env->stones[p] >= 2 && env->scores[p] >= 1);
            } else {
                // Extraction: 2 Crystal, 1 Stone
                affordable = (env->scores[p] >= 2 && env->stones[p] >= 1);
            }
        } else {
            // Street: 1 Crystal + 1 Stone
            affordable = (env->scores[p] >= 1 && env->stones[p] >= 1);
        }
        
        if (!affordable) {
            // Cannot afford - invalid move
            env->rewards[0] += REWARD_INVALID_MOVE;
            env->invalid_placements++;
        } else if (!valid_place || !connected) {
             // Can afford but placement invalid
             env->rewards[0] += REWARD_INVALID_MOVE;
             env->invalid_placements++;
        } else {
             // Valid, connected, and affordable - spend resources and place
             if (card->is_building) {
                 if (card->is_quarry) {
                     env->stones[p] -= 2;
                     env->scores[p] -= 1;
                 } else {
                     env->scores[p] -= 2;
                     env->stones[p] -= 1;
                 }
             } else {
                 // Street: 1 Crystal + 1 Stone
                 env->scores[p] -= 1;
                 env->stones[p] -= 1;
             }
             
             // Success - place tiles
             int type = TILE_STREET;
             if (card->is_building) {
                 type = card->is_quarry ? TILE_QUARRY : TILE_EXTRACTION_BUILDING;
             }
             
             // Place
             for (int k=0; k<card->shape.num_cells; k++) {
                 int bx = x + rotated_shape[k][0];
                 int by = y + rotated_shape[k][1];
                 int idx = by * env->width + bx;
                 env->board_owner[idx] = p + 1;
                 env->board_structure[idx] = type; 
             }
             
             // Track quarry buildings
             if (type == TILE_QUARRY && env->quarry_count[p] < MAX_QUARRY_BUILDINGS_PER_PLAYER) {
                 env->quarry_buildings[p][env->quarry_count[p]].x = x;
                 env->quarry_buildings[p][env->quarry_count[p]].y = y;
                 env->quarry_count[p]++;
             }
             
             env->valid_placements++;
             env->rewards[0] += REWARD_VALID_PLACEMENT;
             
             // Penalty for Quarry near unconnected crystal
             if (type == TILE_QUARRY) {
                  for(int i=0; i<card->shape.num_cells; i++) {
                    int bx = x + rotated_shape[i][0];
                    int by = y + rotated_shape[i][1];
                    for(int c=0; c<NUM_CRYSTALS; c++) {
                         int d = abs(bx - env->crystals[c].x) + abs(by - env->crystals[c].y);
                         if (d == 1) {
                             if (env->crystals[c].connections[p] == 0) {
                                 env->rewards[0] += REWARD_STREET_BLOCKING_PENALTY;
                             }
                         }
                    }
                 }
             }
             
             replace_used_card(env, card_idx);
             update_crystal_connections(env);
             env->consecutive_passes = 0;
        }
    } else {
        // Unknown action
        env->rewards[0] += REWARD_INVALID_MOVE;
    }
    
    // Stats Tracking
    for (int i=0; i<NUM_CRYSTALS; i++) {
        env->acc_p1_crystals_connected_sum += (float)env->crystals[i].connections[0];
        env->acc_p2_crystals_connected_sum += (float)env->crystals[i].connections[1];
        env->acc_p3_crystals_connected_sum += (float)env->crystals[i].connections[2];
        env->acc_p4_crystals_connected_sum += (float)env->crystals[i].connections[3];
    }

    env->current_episode_return += env->rewards[0];
    if (env->current_player == 0) env->current_p1_return += env->rewards[0];
    else if (env->current_player == 1) env->current_p2_return += env->rewards[0];
    else if (env->current_player == 2) env->current_p3_return += env->rewards[0];
    else if (env->current_player == 3) env->current_p4_return += env->rewards[0];
    
    // Consumers action handling
    env->human_action_ready = 0;
    env->last_human_action = 0;
    env->log.human_action_ready = 0;

    // Time & Resource Logic
    env->tick++;
    
    // Count turns for resource distribution
    // This includes both active player turns AND auto-skipped turns of finished/dead players
    // When we advance to the next player, we'll count any skipped players
    env->active_turns++;
    if (env->active_turns % NUM_PLAYERS == 0) {
        distribute_resources(env);
    }
    
    if (pending_rewards[p] > 0) {
        env->rewards[0] += pending_rewards[p];
        pending_rewards[p] = 0;
    }

    // 2. Determine Next Active Player
    int active_count = 0;
    for(int i=0; i<NUM_PLAYERS; i++) {
        if (!env->finished[i] && !env->dead[i]) active_count++;
    }

    int next_p = env->current_player;
    int found_next = 0;
    int skipped_players = 0;
    for(int i=0; i<NUM_PLAYERS; i++) {
        next_p = (next_p + 1) % NUM_PLAYERS;
        if (!env->finished[next_p] && !env->dead[next_p]) {
            env->current_player = next_p;
            found_next = 1;
            break;
        }
        // Count skipped (finished/dead) players
        skipped_players++;
    }
    
    // Add skipped player turns to active_turns
    // This ensures that auto-passing turns of finished/dead players count toward resource distribution
    env->active_turns += skipped_players;
    
    // Check if we should distribute resources after counting skipped players
    if (skipped_players > 0 && env->active_turns % NUM_PLAYERS == 0) {
        distribute_resources(env);
    }
    
    // Check Terminations
    int resources_depleted = 1;
    for(int i=0; i<NUM_CRYSTALS; i++) {
        if (env->crystals[i].resources > 0) { resources_depleted = 0; break; }
    }

    int force_end = 0;
    if (active_count == 0) force_end = 1;
    if (!found_next) force_end = 1; 
    if (env->tick >= env->max_steps) force_end = 1;
    if (resources_depleted) force_end = 1;
    
    if (force_end) {
        env->terminals[0] = 1;
        
        // Final Penalties & Bonuses
        
        // 1. Resource Left Penalty
        for(int i=0; i<NUM_PLAYERS; i++) {
            float penalty = (env->scores[i] + env->stones[i]) * REWARD_RESOURCE_LEFT_PENALTY;
            if (i==0) env->current_p1_return += penalty;
            else if (i==1) env->current_p2_return += penalty;
            else if (i==2) env->current_p3_return += penalty;
            else if (i==3) env->current_p4_return += penalty;
        }

        // 2. Win Bonus (Based on Cumulative Extracted)
        int current_price = 0;
        int current_max_score = -1;
        int current_best_player = -1;
        bool players_got_reward[NUM_PLAYERS] = {false, false, false, false};

        // Rank loop
        for (int p=0; p<NUM_PLAYERS; p++) {
            if (env->cumulative_crystals[p] > current_max_score) {
                current_max_score = env->cumulative_crystals[p];
                current_best_player = p;
            }
        }

        while (current_best_player != -1) {
            int number_of_tied_players = 0;
            for (int p=0; p<NUM_PLAYERS; p++) {
                if (env->cumulative_crystals[p] == current_max_score) {
                    number_of_tied_players++;
                }
            }
            int adjusted_price = current_price;
            while (number_of_tied_players > 1) {
                number_of_tied_players--;
                adjusted_price++;
            } 

            float bonus = REWARD_WIN_BONUS[adjusted_price];
            
            // Distribute bonus to all tied players
            for (int p=0; p<NUM_PLAYERS; p++) {
                 if (!players_got_reward[p] && env->cumulative_crystals[p] == current_max_score) {
                     players_got_reward[p] = true;
                     if(p==0) env->current_p1_return += bonus;
                     else if(p==1) env->current_p2_return += bonus;
                     else if(p==2) env->current_p3_return += bonus;
                     else if(p==3) env->current_p4_return += bonus;
                     
                     if (p == env->current_player) {
                         env->rewards[0] += bonus;
                     }
                 }
            }

            current_price++; // Next price tier
            
            current_best_player = -1;
            current_max_score = -1;
            for (int p=0; p<NUM_PLAYERS; p++) {
                if (players_got_reward[p]) continue;
                if (env->cumulative_crystals[p] > current_max_score) {
                    current_max_score = env->cumulative_crystals[p];
                    current_best_player = p;
                }
            }
        }

        // Logging
        env->log.n = 1.0f;
        env->log.episode_length = (float)env->tick;
        env->log.episode_return = env->current_episode_return;
        env->log.p1_return = env->current_p1_return;
        env->log.p2_return = env->current_p2_return;
        env->log.p3_return = env->current_p3_return;
        env->log.p4_return = env->current_p4_return;
        
        env->log.p1_crystal_collected = env->acc_p1_crystal_collected;
        env->log.p2_crystal_collected = env->acc_p2_crystal_collected;
        env->log.p3_crystal_collected = env->acc_p3_crystal_collected;
        env->log.p4_crystal_collected = env->acc_p4_crystal_collected;
        
        env->log.p1_stone_collected = env->acc_p1_stone_collected;
        env->log.p2_stone_collected = env->acc_p2_stone_collected;
        env->log.p3_stone_collected = env->acc_p3_stone_collected;
        env->log.p4_stone_collected = env->acc_p4_stone_collected;
        
        float steps = (float)env->tick;
        if (steps < 1.0f) steps = 1.0f;
        env->log.p1_crystals_connected = env->acc_p1_crystals_connected_sum / steps;
        env->log.p2_crystals_connected = env->acc_p2_crystals_connected_sum / steps;
        env->log.p3_crystals_connected = env->acc_p3_crystals_connected_sum / steps;
        env->log.p4_crystals_connected = env->acc_p4_crystals_connected_sum / steps;
        
        env->log.crystals_connected = (env->log.p1_crystals_connected + env->log.p2_crystals_connected + 
                                       env->log.p3_crystals_connected + env->log.p4_crystals_connected) / 4.0f;
        
        // Win Rate
        int max_sc = -1;
        for (int p=0; p<NUM_PLAYERS; p++) if (env->cumulative_crystals[p] > max_sc) max_sc = env->cumulative_crystals[p];
        env->log.p1_win_rate = (env->cumulative_crystals[0] == max_sc) ? 1.0f : 0.0f;
        env->log.p2_win_rate = (env->cumulative_crystals[1] == max_sc) ? 1.0f : 0.0f;
        env->log.p3_win_rate = (env->cumulative_crystals[2] == max_sc) ? 1.0f : 0.0f;
        env->log.p4_win_rate = (env->cumulative_crystals[3] == max_sc) ? 1.0f : 0.0f;
    }
    
    // 8. Obs
    if (!env->terminals[0]) {
        fill_observation_comp(env);
    }
}

void q_close(Quarryboard* env) {
    if(env->board_owner) free(env->board_owner);
    if(env->board_structure) free(env->board_structure);
    if(env->board_feature) free(env->board_feature);
}


// c_render moved below


// ============================================================================
// Standalone / Verify Mode
// ============================================================================

#define MAX_TEST_BOARD 80
#define MAX_OBS_SIZE (MAX_TEST_BOARD*MAX_TEST_BOARD + 50000) 

static float observations[MAX_OBS_SIZE];
static int actions[1];
static float rewards[1];
static float total_reward[NUM_PLAYERS];
static unsigned char terminals[1];




// Player Colors (Brighter for borders)
static const Color PLAYER_COLORS[4] = {
    {50, 50, 255, 255},   // P1 Blue
    {255, 50, 50, 255},   // P2 Red
    {50, 255, 50, 255},   // P3 Green
    {255, 255, 50, 255}   // P4 Yellow
};

// CLI Render Mode - Text-based output for testing
void q_render_cli(Quarryboard* env) {
    printf("\n========================================\n");
    printf("QUARRYBOARD - Tick %d / %d (Round %d)\n", env->tick, env->max_steps, env->tick/4);
    printf("========================================\n\n");
    
    // Current Player
    printf(">>> CURRENT PLAYER: P%d <<<\n\n", env->current_player + 1);
    
    // Player Resources
    printf("PLAYER RESOURCES:\n");
    for (int i = 0; i < NUM_PLAYERS; i++) {
        printf("  P%d: %2dC %2dS | Cumulative: %2d | ", 
               i+1, env->scores[i], env->stones[i], env->cumulative_crystals[i]);
        if (env->finished[i]) printf("FINISHED");
        else if (env->dead[i]) printf("DEAD");
        else printf("ACTIVE");
        printf("\n");
    }
    printf("\n");
    
    // Market
    printf("MARKET:\n");
    printf("  Buildings (0-3):\n");
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        Card* card = &env->building_market[i];
        char type = card->is_quarry ? 'Q' : 'E';
        int cost_c = card->is_quarry ? 1 : 2;
        int cost_s = card->is_quarry ? 2 : 1;
        printf("    [%d] %c (cells=%d, cost=%dC+%dS)\n", 
               i, type, card->shape.num_cells, cost_c, cost_s);
    }
    printf("  Streets (4-6):\n");
    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        Card* card = &env->street_market[i];
        printf("    [%d] S (cells=%d, cost=1C+1S)\n", 
               i+4, card->shape.num_cells);
    }
    printf("\n");
    
    // Crystals
    printf("CRYSTAL NODES:\n");
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        printf("  C%d (%2d,%2d): %2d res | Connections: ", 
               i, env->crystals[i].x, env->crystals[i].y, env->crystals[i].resources);
        for (int p = 0; p < NUM_PLAYERS; p++) {
            printf("P%d=%d ", p+1, env->crystals[i].connections[p]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Board Summary
    int tile_counts[NUM_PLAYERS][6] = {0}; // [player][tile_type] - expanded to 6 for safety
    for (int i = 0; i < env->board_tiles; i++) {
        int owner = env->board_owner[i];
        if (owner > 0) {
            int type = env->board_structure[i];
            if (type < 6) { // Safety check
                tile_counts[owner-1][type]++;
            }
        }
    }
    
    printf("BOARD SUMMARY:\n");
    for (int p = 0; p < NUM_PLAYERS; p++) {
        printf("  P%d: %d Streets, %d Extractors, %d Quarries, %d Bases\n",
               p+1, 
               tile_counts[p][TILE_STREET],
               tile_counts[p][TILE_EXTRACTION_BUILDING],
               tile_counts[p][TILE_QUARRY],
               tile_counts[p][TILE_START_BASE]);
    }
    printf("\n");
    
    if (env->terminals[0]) {
        printf("*** GAME OVER ***\n");
        int max_sc = -1;
        for (int p = 0; p < NUM_PLAYERS; p++) {
            if (env->cumulative_crystals[p] > max_sc) max_sc = env->cumulative_crystals[p];
        }
        printf("Winner(s): ");
        for (int p = 0; p < NUM_PLAYERS; p++) {
            if (env->cumulative_crystals[p] == max_sc) printf("P%d ", p+1);
        }
        printf("(Score: %d)\n\n", max_sc);
    }
}

void q_draw_internal(Quarryboard* env) {
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim;
    if (cell_size > 48) cell_size = 48;
    if (cell_size < 8) cell_size = 8;
    
    // Draw grid
    for (int i = 0; i <= env->width; i++) {
        DrawLine(i * cell_size, 0, i * cell_size, env->height * cell_size, PUFF_GRID);
    }
    for (int i = 0; i <= env->height; i++) {
        DrawLine(0, i * cell_size, env->width * cell_size, i * cell_size, PUFF_GRID);
    }
    
    // Draw objects
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            int px = x * cell_size;
            int py = y * cell_size;
            
            // Feature (Crystals)
            if (env->board_feature[idx] == TILE_CRYSTAL_NODE) {
                // Find which crystal
                int res = 0;
                for(int k=0; k<NUM_CRYSTALS; k++) {
                   if(env->crystals[k].x == x && env->crystals[k].y == y) {
                       res = env->crystals[k].resources;
                       break;
                   }
                }
                Color c = (res > 0) ? PUFF_PURPLE : PUFF_GRID;
                DrawCircle(px + cell_size/2, py + cell_size/2, cell_size/3, c);
                if (cell_size >= 20 && res > 0) {
                     char buf[8];
                     sprintf(buf, "%d", res);
                     DrawText(buf, px + cell_size/2 - 4, py + cell_size/2 - 4, 10, PUFF_WHITE);
                }
            }
            
            // Structure
            unsigned char owner = env->board_owner[idx]; // 1-based
            unsigned char type = env->board_structure[idx];
            
            if (owner > 0 && type != TILE_EMPTY) {
                Color border_color = PLAYER_COLORS[owner - 1];
                Color fill_color = PUFF_GRID; // Default
                
                if (type == TILE_STREET) {
                    fill_color = PUFF_CYAN;
                    fill_color.a = 200; // Slightly transparent
                    
                    DrawRectangle(px + cell_size/4, py + cell_size/4, cell_size/2, cell_size/2, fill_color);
                    DrawRectangleLinesEx((Rectangle){px + cell_size/4, py + cell_size/4, cell_size/2, cell_size/2}, 2, border_color);
                    
                } else if (type == TILE_EXTRACTION_BUILDING) {
                    fill_color = PUFF_PURPLE;
                    fill_color.a = 200;
                    
                    DrawRectangle(px + 2, py + 2, cell_size - 4, cell_size - 4, fill_color);
                    DrawRectangleLinesEx((Rectangle){px + 2, py + 2, cell_size - 4, cell_size - 4}, 2, border_color);
                    
                } else if (type == TILE_QUARRY) {
                    fill_color = PUFF_GREEN;
                    fill_color.a = 200;
                    
                    DrawRectangle(px + 2, py + 2, cell_size - 4, cell_size - 4, fill_color);
                    DrawRectangleLinesEx((Rectangle){px + 2, py + 2, cell_size - 4, cell_size - 4}, 2, border_color);
                    
                } else if (type == TILE_START_BASE) {
                    fill_color = PUFF_WHITE;
                    fill_color.a = 150;
                    
                    DrawRectangle(px + 1, py + 1, cell_size - 2, cell_size - 2, fill_color);
                    DrawRectangleLinesEx((Rectangle){px + 1, py + 1, cell_size - 2, cell_size - 2}, 3, border_color);
                    DrawText("B", px + cell_size/2 - 4, py + cell_size/2 - 4, 10, PUFF_BACKGROUND);
                }
            }
        }
    }
}

static void draw_shape_preview(int x, int y, Card* card, int rotation, Color color, int cell_size) {
    int rotated_shape[MAX_SHAPE_SIZE][2];
    int raw_shape[MAX_SHAPE_SIZE][2];
    for (int i = 0; i < card->shape.num_cells; i++) {
        raw_shape[i][0] = card->shape.cells[i][0];
        raw_shape[i][1] = card->shape.cells[i][1];
    }
    rotate_shape(raw_shape, card->shape.num_cells, rotation, rotated_shape);
    
    for (int i = 0; i < card->shape.num_cells; i++) {
        int px = x + rotated_shape[i][0] * cell_size;
        int py = y + rotated_shape[i][1] * cell_size;
        DrawRectangle(px + 1, py + 1, cell_size - 2, cell_size - 2, color);
    }
}

static void draw_market(Quarryboard* env, int cell_size, int start_x, int selected_card) {
    int padding = 10;
    int current_y = padding;
    
    DrawText("MARKET", start_x + 10, current_y, 20, PUFF_WHITE);
    current_y += 30;
    
    for (int i = 0; i < TOTAL_MARKET_SLOTS; i++) {
        Card* card;
        Color color;
        
        if (i < NUM_BUILDING_SLOTS) {
             card = &env->building_market[i];
             // Different colors for Quarries vs Extractors
             color = card->is_quarry ? PUFF_GREEN : PUFF_PURPLE;
        } else {
             card = &env->street_market[i - NUM_BUILDING_SLOTS];
             color = PUFF_CYAN;
        }
        
        // Highlight selection
        if (i == selected_card) {
             DrawRectangle(start_x, current_y - 5, 180, 5 * cell_size + 10, (Color){255, 255, 255, 50});
             DrawRectangleLines(start_x, current_y - 5, 180, 5 * cell_size + 10, PUFF_YELLOW);
        }
        
        // Draw Card Shape
        draw_shape_preview(start_x + 20, current_y, card, 0, color, cell_size);
        
        // Draw Key Hint
        DrawText(TextFormat("[%d]", i), start_x + 5, current_y + 10, 10, PUFF_WHITE);
        
        // Draw Cost
        if (i < NUM_BUILDING_SLOTS) {
            // Building: either Quarry or Extractor
            if (card->is_quarry) {
                DrawText("1C+2S", start_x + 90, current_y + 5, 12, PUFF_WHITE);
            } else {
                DrawText("2C+1S", start_x + 90, current_y + 5, 12, PUFF_WHITE);
            }
        } else {
            // Street
            DrawText("1C+1S", start_x + 90, current_y + 5, 12, PUFF_WHITE);
        }
        
        current_y += 5 * cell_size + 15;
    }
}


// SetTargetFPS(0); // Disable FPS limiting to avoid hang on Mac M4

void q_render(Quarryboard* env) {
    // CLI Mode - text output only
    if (env->render_mode == 2) {
        q_render_cli(env);
        return;
    }
    
    if (!IsWindowReady()) {
        int max_dim = (env->width > env->height) ? env->width : env->height;
        int cell_size = 800 / max_dim;
        if (cell_size > 48) cell_size = 48;
        if (cell_size < 8) cell_size = 8;
        int market_width = 200;
        int win_w = env->width * cell_size + market_width;
        int win_h = env->height * cell_size + 100;
        InitWindow(win_w, win_h, "Quarryboard (Eval)");
        
        if (env->render_fps == 0) env->render_fps = 60;
        // SetTargetFPS(env->render_fps); 
        // NOTE: SetTargetFPS(60) freezes on Mac M4. Rely on VSync or Python pacing.
        
        // Ensure inputs are fresh
        env->cursor_x = env->width / 2;
        env->cursor_y = env->height / 2;
        env->selected_card = 0;
        env->rotation = 0;
    }
    
    // Explicitly poll input events to keep window responsive and capture input
    // This is crucial when calling render frequenty from Python loop
    PollInputEvents();
    
    // FPS Control
    if (IsKeyPressed(KEY_LEFT_BRACKET)) {
        env->render_fps = (env->render_fps > 10) ? env->render_fps - 10 : 1;
        // SetTargetFPS(env->render_fps);
    }
    if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
        env->render_fps += 10;
        // SetTargetFPS(env->render_fps);
    }
    
    // --- Input Handling ---
    if (!env->terminals[0]) {
        if (!env->human_action_ready) {
             for (int i = 0; i < TOTAL_MARKET_SLOTS; i++) {
                 if (IsKeyPressed(KEY_ZERO + i) || IsKeyPressed(KEY_KP_0 + i)) env->selected_card = i;
             }
             if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) env->cursor_x = (env->cursor_x - 1 + env->width) % env->width;
             if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) env->cursor_x = (env->cursor_x + 1) % env->width;
             if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) env->cursor_y = (env->cursor_y - 1 + env->height) % env->height;
             if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) env->cursor_y = (env->cursor_y + 1) % env->height;
             if (IsKeyPressed(KEY_Q)) env->rotation = (env->rotation - 1 + 4) % 4;
             if (IsKeyPressed(KEY_E)) env->rotation = (env->rotation + 1) % 4;
             
             if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
                 // Encode Action
                 int m_action = encode_action_comp(env, env->selected_card, env->cursor_x, env->cursor_y, env->rotation);
                 env->last_human_action = m_action;
                 env->human_action_ready = 1;
                 
                 // Expose to Log immediately
                 env->log.human_action_ready = 1.0f;
                 env->log.last_human_action = (float)m_action;
             }
             
             // Pass Action (P key)
             if (IsKeyPressed(KEY_P)) {
                 int num_std_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
                 int pass_action = num_std_actions;
                 env->last_human_action = pass_action;
                 env->human_action_ready = 1;
                 
                 // Expose to Log immediately
                 env->log.human_action_ready = 1.0f;
                 env->log.last_human_action = (float)pass_action;
             }
             
             // Finish Action (F key)
             if (IsKeyPressed(KEY_F)) {
                 int num_std_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
                 int finish_action = num_std_actions + 1;
                 env->last_human_action = finish_action;
                 env->human_action_ready = 1;
                 
                 // Expose to Log immediately
                 env->log.human_action_ready = 1.0f;
                 env->log.last_human_action = (float)finish_action;
             }
        }
        
        // Update Observation Buffer In-Place for fast Python polling
        // Offset = Board (W*H) + Market (210) + Crystals (70) + Resources (20)
        int obs_offset = env->board_tiles + 210 + 70 + 20;
        env->observations[obs_offset + 0] = (float)env->current_player;
        env->observations[obs_offset + 1] = (float)env->human_action_ready;
        env->observations[obs_offset + 2] = (float)env->last_human_action;
        
        // Always update current player in log for Python polling
        env->log.current_player = (float)env->current_player;
        
    } else {
        env->human_action_ready = 0;
        env->log.human_action_ready = 0.0f;
    }
    
    // Calculate cell_size
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim;
    if (cell_size > 48) cell_size = 48;
    if (cell_size < 8) cell_size = 8;
    
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    q_draw_internal(env);
    
    // Draw Market
    draw_market(env, cell_size, env->width * cell_size + 10, env->selected_card);
    
    // Draw Cursor
    int cx = env->cursor_x * cell_size;
    int cy = env->cursor_y * cell_size;
    DrawRectangleLines(cx, cy, cell_size, cell_size, PUFF_WHITE);
    
    // Draw Preview
    Card* card = (env->selected_card < NUM_BUILDING_SLOTS) 
                ? &env->building_market[env->selected_card]
                : &env->street_market[env->selected_card - NUM_BUILDING_SLOTS];
    
    Color preview_color;
    if (env->selected_card < NUM_BUILDING_SLOTS) {
        // Building: green for quarry, purple for extractor
        preview_color = card->is_quarry ? PUFF_GREEN : PUFF_PURPLE;
    } else {
        // Street: cyan
        preview_color = PUFF_CYAN;
    }
    preview_color.a = 150;
    draw_shape_preview(cx, cy, card, env->rotation, preview_color, cell_size);
    
    // Draw HUD
    int y_hud = env->height * cell_size + 10;
    DrawText(TextFormat("Tick: %d / %d | Round: %d", env->tick, env->max_steps, env->tick/4), 10, y_hud, 20, PUFF_WHITE);
    DrawText(TextFormat("Current Player: P%d", env->current_player+1), 10, y_hud+25, 20, PLAYER_COLORS[env->current_player]);
    
    for(int i=0; i<4; i++) {
        DrawText(TextFormat("P%d: %dC %dS", i+1, env->scores[i], env->stones[i]), 250 + i*120, y_hud, 20, PLAYER_COLORS[i]);
    }
    
    // Controls help text
    DrawText("Controls: 0-6=Card | WASD/Arrows=Move | Q/E=Rotate | Space=Place | P=Pass | F=Finish", 10, y_hud+50, 16, PUFF_CYAN);
    
    EndDrawing();
    
    // Check if game over, then pause
    if (env->terminals[0]) {
        BeginDrawing();
        // Draw Overlay to indicate end
        int win_w = env->width * cell_size + 200;
        int win_h = env->height * cell_size + 100;
        DrawRectangle(0, win_h / 2 - 40, win_w, 80, (Color){0, 0, 0, 200});
        DrawText("GAME OVER", win_w/2 - 80, win_h/2 - 20, 30, PUFF_WHITE);
        char cause[64];
        if (env->tick >= env->max_steps) snprintf(cause, 64, "Max Steps Reached");
        else snprintf(cause, 64, "Crystals Empty or Deadlock");
        DrawText(cause, win_w/2 - MeasureText(cause, 20)/2, win_h/2 + 15, 20, PUFF_WHITE);
        EndDrawing();
    }
}

void play_with_render(Quarryboard* env) {
    int selected_card = 0;
    int cursor_x = env->width / 2;
    int cursor_y = env->height / 2;
    int rotation = 0;
    
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim;
    if (cell_size > 48) cell_size = 48;
    if (cell_size < 8) cell_size = 8;
    
    int market_width = 200;
    int win_w = env->width * cell_size + market_width;
    int win_h = env->height * cell_size + 100;
    
    if (!IsWindowReady()) {
        InitWindow(win_w, win_h, "Crystalboard Comp 4-Player");
        SetTargetFPS(60);
    }
    
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
             q_reset(env);
             env->rewards[0] = 0;
             env->terminals[0] = 0;
             for(int i=0; i<4; i++) total_reward[i] = 0;
        }
        
        // Input only if not terminal
        if (!env->terminals[0]) {
            for (int i = 0; i < TOTAL_MARKET_SLOTS; i++) {
                if (IsKeyPressed(KEY_ZERO + i) || IsKeyPressed(KEY_KP_0 + i)) selected_card = i;
            }
            if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) cursor_x = (cursor_x - 1 + env->width) % env->width;
            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) cursor_x = (cursor_x + 1) % env->width;
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) cursor_y = (cursor_y - 1 + env->height) % env->height;
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) cursor_y = (cursor_y + 1) % env->height;
            if (IsKeyPressed(KEY_Q)) rotation = (rotation - 1 + 4) % 4;
            if (IsKeyPressed(KEY_E)) rotation = (rotation + 1) % 4;
            
            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
                // Current player acts
                int p = env->current_player;
                env->actions[0] = encode_action_comp(env, selected_card, cursor_x, cursor_y, rotation);
                q_step(env); // Advances turn
                total_reward[p] += env->rewards[0];
            }
            
            // Pass Action (P key)
            if (IsKeyPressed(KEY_P)) {
                int p = env->current_player;
                int num_std_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
                env->actions[0] = num_std_actions; // Pass action
                q_step(env);
                total_reward[p] += env->rewards[0];
            }
            
            // Finish Action (F key)
            if (IsKeyPressed(KEY_F)) {
                int p = env->current_player;
                int num_std_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
                env->actions[0] = num_std_actions + 1; // Finish action
                q_step(env);
                total_reward[p] += env->rewards[0];
            }
        }
        
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);
        q_draw_internal(env);
        
        // Draw Market
        draw_market(env, cell_size, env->width * cell_size + 10, selected_card);
        
        // Draw Cursor
        int cx = cursor_x * cell_size;
        int cy = cursor_y * cell_size;
        DrawRectangleLines(cx, cy, cell_size, cell_size, PUFF_WHITE);
        
        // Draw Preview
        Card* card = (selected_card < NUM_BUILDING_SLOTS) 
                    ? &env->building_market[selected_card]
                    : &env->street_market[selected_card - NUM_BUILDING_SLOTS];
        
        Color preview_color;
        if (selected_card < NUM_BUILDING_SLOTS) {
            preview_color = card->is_quarry ? PUFF_GREEN : PUFF_PURPLE;
        } else {
            preview_color = PUFF_CYAN;
        }
        preview_color.a = 150;
        draw_shape_preview(cx, cy, card, rotation, preview_color, cell_size);
        
        // HUD
        int y_hud = env->height * cell_size + 10;
        DrawText(TextFormat("Tick: %d / %d | Round: %d", env->tick, env->max_steps, env->tick/4), 10, y_hud, 20, PUFF_WHITE);
        DrawText(TextFormat("Current Player: P%d", env->current_player+1), 10, y_hud+25, 20, PLAYER_COLORS[env->current_player]);
        
        for(int i=0; i<4; i++) {
             DrawText(TextFormat("P%d: %dC %dS", i+1, env->scores[i], env->stones[i]), 250 + i*120, y_hud, 20, PLAYER_COLORS[i]);
        }
        
        if (env->terminals[0]) {
             DrawText("GAME OVER", win_w/2 - 150, win_h/2, 40, PUFF_RED);
        }
        
        EndDrawing();
    }
}

#ifndef PUFFER_BINDING
int main(int argc, char* argv[]) {
    Quarryboard env = {0};
    int seed = 42;
    int board_size = 30; // Competitive map default
    int render_mode = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--render") == 0 || strcmp(argv[i], "-r") == 0) render_mode = 1;
        else if (i==1) seed = atoi(argv[i]);
    }
    
    env.width = board_size;
    env.height = board_size;
    env.board_tiles = board_size*board_size;
    
    // Alloc
    env.board_owner = (unsigned char*)calloc(env.board_tiles, 1);
    env.board_structure = (unsigned char*)calloc(env.board_tiles, 1);
    env.board_feature = (unsigned char*)calloc(env.board_tiles, 1);
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.seed = seed;
    env.max_steps = 1000; 
    
    q_reset(&env);
    
    if (render_mode) {
        printf("Starting 4-Player Crystalboard Competitive...\n");
        printf("Use NUMBER keys to select card, Q/E to rotate, SPACE to place.\n");
        printf("Press P to Pass, F to Finish your turn.\n");
        printf("You control ALL players in turn.\n");
        play_with_render(&env);
    } else {
        printf("Console mode not fully implemented for verification. Use --render.\n");
    }
    
    q_close(&env);
    return 0;
}
#endif
