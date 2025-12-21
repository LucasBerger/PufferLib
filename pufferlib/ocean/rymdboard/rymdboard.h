/* Rymdboard: A tile-placement game environment for PufferLib Ocean
 * 
 * The goal is to connect crystal nodes on a 10x10 grid by placing
 * streets and extraction buildings from a market of 7 cards.
 * Win condition: Connect 2+ crystal nodes with extraction buildings.
 * 
 * Reference: research/simple_env.py, research/simple_game_engine.py
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "raylib.h"

// Board dimensions
#define BOARD_SIZE 10
#define BOARD_TILES (BOARD_SIZE * BOARD_SIZE)

// Market sizes
#define NUM_BUILDING_SLOTS 4
#define NUM_STREET_SLOTS 3
#define TOTAL_MARKET_SLOTS 7

// Action space: 7 cards × 10 × 10 positions × 4 rotations = 2800
#define NUM_ACTIONS 2800

// Crystal nodes
#define NUM_CRYSTALS 3

// Max shapes
#define MAX_SHAPE_SIZE 5

// Tile types (matches TileType enum in game/tile.py)
#define TILE_EMPTY 0
#define TILE_STREET 1
#define TILE_EXTRACTION_BUILDING 2
#define TILE_START_BASE 3
#define TILE_CRYSTAL_NODE 4

// Shape definitions (common shapes from game/default_content.py)
// Each shape is an array of (dx, dy) offsets, terminated by (-1, -1)
typedef struct {
    int cells[MAX_SHAPE_SIZE][2];  // Up to 5 cells per shape
    int num_cells;
} Shape;

// Card types
typedef struct {
    Shape shape;
    int is_building;  // 1 for building, 0 for street
    int static_points;
} Card;

// Crystal node info
typedef struct {
    int x;
    int y;
    int amount;
    int connected;  // 1 if adjacent to extraction building
} CrystalNode;

// Required struct for logging (only use floats!)
typedef struct {
    float perf;             // Normalized performance (0-1)
    float score;            // Unnormalized score
    float episode_return;   // Sum of rewards
    float episode_length;   // Number of steps
    float crystals_connected;
    float valid_placements;
    float invalid_placements;
    float n;                // Required as last field
} Log;

// Main environment struct
typedef struct {
    Log log;                          // Required field for logging
    
    // Required buffers (set by Python binding)
    float* observations;              // Observation buffer
    int* actions;                     // Action buffer (discrete)
    float* rewards;                   // Reward buffer
    unsigned char* terminals;         // Terminal flag buffer
    
    // Game state
    unsigned char board_owner[BOARD_TILES];      // 0=empty, 1=player
    unsigned char board_structure[BOARD_TILES];  // Tile type
    unsigned char board_feature[BOARD_TILES];    // Features like crystal nodes
    
    // Market state - cards are drawn from decks and replaced when used
    Card building_market[NUM_BUILDING_SLOTS];
    Card street_market[NUM_STREET_SLOTS];
    // Available cards in decks (shapes that can be drawn)
    int building_deck[21];  // 3 copies of each of 7 shapes = 21 cards
    int street_deck[21];    // Same for streets
    int building_deck_size;
    int street_deck_size;
    
    // Crystal nodes
    CrystalNode crystals[NUM_CRYSTALS];
    
    // Episode state
    int tick;
    int max_steps;
    int seed;
    unsigned int rng_state;    // RNG state that persists and evolves
    unsigned int reset_count;  // Counter for number of resets
    
    // Stats for this episode
    int valid_placements;
    int invalid_placements;
    int crystals_connected;

    int frameskip;
    int render_mode;
} Rymdboard;

// Reward constants (from research/reward_constants.py)
// Invalid moves now terminate the episode, so penalty doesn't need to be extreme
#define REWARD_STEP_PENALTY -0.01f
#define REWARD_INVALID_MOVE -1.0f
#define REWARD_VALID_PLACEMENT 0.5f
#define REWARD_DISTANCE_SCALE 0.5f
#define REWARD_CONNECTION 10.0f
#define REWARD_WIN_BONUS 50.0f
#define REWARD_NO_VALID_ACTIONS -1.0f

// Shape library - common shapes from game/default_content.py
static const int SHAPE_L[4][2] = {{0,0}, {0,1}, {0,2}, {1,2}};
static const int SHAPE_L_SIZE = 4;

static const int SHAPE_T[4][2] = {{1,0}, {0,1}, {1,1}, {1,2}};
static const int SHAPE_T_SIZE = 4;

static const int SHAPE_LITTLE_L[3][2] = {{0,0}, {1,0}, {0,1}};
static const int SHAPE_LITTLE_L_SIZE = 3;

static const int SHAPE_Z[4][2] = {{1,0}, {2,0}, {0,1}, {1,1}};
static const int SHAPE_Z_SIZE = 4;

static const int SHAPE_I[4][2] = {{0,0}, {1,0}, {2,0}, {3,0}};
static const int SHAPE_I_SIZE = 4;

static const int SHAPE_LITTLE_I[3][2] = {{0,0}, {1,0}, {2,0}};
static const int SHAPE_LITTLE_I_SIZE = 3;

static const int SHAPE_BOX[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};
static const int SHAPE_BOX_SIZE = 4;

// Start base shape (2x2)
static const int SHAPE_BASE[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};
static const int SHAPE_BASE_SIZE = 4;

// Function declarations
void add_log(Rymdboard* env);
void c_reset(Rymdboard* env);
void c_step(Rymdboard* env);
void c_render(Rymdboard* env);
void c_close(Rymdboard* env);

// Helper function forward declarations (only those needed before definition)
static void rotate_shape(const int shape[][2], int num_cells, int rotation, int out_shape[][2]);

// ============================================================================
// Implementation
// ============================================================================

static unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Shape library arrays for easy indexing
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

// Initialize card decks - multiple copies of each shape as in default_content.py
static void init_card_decks(Rymdboard* env) {
    // Building deck: 3 copies of each of the 7 shapes = 21 cards
    env->building_deck_size = 21;
    int deck_idx = 0;
    for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
        for (int copy = 0; copy < 3; copy++) {
            env->building_deck[deck_idx++] = shape;
        }
    }

    // Street deck: same structure
    env->street_deck_size = 21;
    deck_idx = 0;
    for (int shape = 0; shape < NUM_SHAPE_TYPES; shape++) {
        for (int copy = 0; copy < 3; copy++) {
            env->street_deck[deck_idx++] = shape;
        }
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
        // If deck is empty, refill it (shouldn't happen in normal play)
        *deck_size = 21;
        for (int i = 0; i < 21; i++) {
            deck[i] = i % 7;  // 3 copies of each shape
        }
        shuffle_deck(deck, *deck_size, rng_state);
    }

    // Draw a random card from deck
    int deck_idx = xorshift32(rng_state) % *deck_size;
    int shape_idx = deck[deck_idx];

    // Remove from deck by swapping with last element
    deck[deck_idx] = deck[*deck_size - 1];
    (*deck_size)--;

    // Initialize card
    copy_shape(card, shape_idx);
    card->is_building = is_building;
    if (is_building) {
        card->static_points = (card->shape.num_cells == 3) ? 3 : 2;
    } else {
        card->static_points = 1;
    }
}

static void replace_used_card(Rymdboard* env, int card_idx) {
    if (card_idx < NUM_BUILDING_SLOTS) {
        // Replace building card
        draw_random_card(&env->building_market[card_idx], env->building_deck,
                        &env->building_deck_size, 1, &env->rng_state);
    } else {
        // Replace street card
        int street_idx = card_idx - NUM_BUILDING_SLOTS;
        draw_random_card(&env->street_market[street_idx], env->street_deck,
                        &env->street_deck_size, 0, &env->rng_state);
    }
}

static void init_decks(Rymdboard* env) {
    // Initialize the card decks
    init_card_decks(env);

    // Shuffle decks
    shuffle_deck(env->building_deck, env->building_deck_size, &env->rng_state);
    shuffle_deck(env->street_deck, env->street_deck_size, &env->rng_state);

    // Draw initial market cards
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        draw_random_card(&env->building_market[i], env->building_deck,
                        &env->building_deck_size, 1, &env->rng_state);
    }

    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        draw_random_card(&env->street_market[i], env->street_deck,
                        &env->street_deck_size, 0, &env->rng_state);
    }
}

static void place_start_base(Rymdboard* env) {
    // Place 2x2 start base at position (1, 1)
    int x = 1, y = 1;
    for (int i = 0; i < SHAPE_BASE_SIZE; i++) {
        int bx = x + SHAPE_BASE[i][0];
        int by = y + SHAPE_BASE[i][1];
        int idx = by * BOARD_SIZE + bx;
        env->board_owner[idx] = 1;  // Player owns it
        env->board_structure[idx] = TILE_START_BASE;
    }
}

static void place_crystal_nodes(Rymdboard* env) {
    // Place crystal nodes in 3 quadrants (excluding top-left where base is)
    // Top-right: x=[5,9], y=[0,4]
    // Bottom-left: x=[0,4], y=[5,9]
    // Bottom-right: x=[5,9], y=[5,9]
    
    // Use rng_state (updated each episode) for variety
    
    // Top-right
    env->crystals[0].x = 5 + (xorshift32(&env->rng_state) % 5);
    env->crystals[0].y = xorshift32(&env->rng_state) % 5;
    env->crystals[0].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[0].connected = 0;
    
    // Bottom-left
    env->crystals[1].x = xorshift32(&env->rng_state) % 5;
    env->crystals[1].y = 5 + (xorshift32(&env->rng_state) % 5);
    env->crystals[1].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[1].connected = 0;
    
    // Bottom-right
    env->crystals[2].x = 5 + (xorshift32(&env->rng_state) % 5);
    env->crystals[2].y = 5 + (xorshift32(&env->rng_state) % 5);
    env->crystals[2].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[2].connected = 0;
    
    // Place on board feature grid
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        int idx = env->crystals[i].y * BOARD_SIZE + env->crystals[i].x;
        env->board_feature[idx] = TILE_CRYSTAL_NODE;
    }
}

static void rotate_shape(const int shape[][2], int num_cells, int rotation, int out_shape[][2]) {
    // Rotate shape 90 degrees clockwise 'rotation' times
    // (x, y) -> (y, -x) per rotation, then normalize
    
    int temp[MAX_SHAPE_SIZE][2];
    
    // Copy input
    for (int i = 0; i < num_cells; i++) {
        temp[i][0] = shape[i][0];
        temp[i][1] = shape[i][1];
    }
    
    for (int r = 0; r < (rotation % 4); r++) {
        for (int i = 0; i < num_cells; i++) {
            int x = temp[i][0];
            int y = temp[i][1];
            temp[i][0] = y;
            temp[i][1] = -x;
        }
    }
    
    // Normalize to start at (0, 0)
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

static int can_place_tile(Rymdboard* env, int x, int y, int shape[][2], int num_cells) {
    // Check bounds and overlap
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        
        // Check bounds
        if (bx < 0 || bx >= BOARD_SIZE || by < 0 || by >= BOARD_SIZE) {
            return 0;
        }
        
        int idx = by * BOARD_SIZE + bx;
        
        // Check overlap with existing structures
        if (env->board_owner[idx] != 0) {
            return 0;
        }
        
        // Check overlap with features (can't place on crystal nodes)
        if (env->board_feature[idx] != TILE_EMPTY) {
            return 0;
        }
    }
    
    return 1;
}

static int is_connected_to_network(Rymdboard* env, int x, int y, int shape[][2], int num_cells) {
    // Check if any cell of the shape is adjacent to player's street or base
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        
        for (int d = 0; d < 4; d++) {
            int nx = bx + dx[d];
            int ny = by + dy[d];
            
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) {
                continue;
            }
            
            int idx = ny * BOARD_SIZE + nx;
            
            // Must be owned by player and be street or base
            if (env->board_owner[idx] == 1) {
                if (env->board_structure[idx] == TILE_STREET ||
                    env->board_structure[idx] == TILE_START_BASE) {
                    return 1;
                }
            }
        }
    }
    
    return 0;
}

static void place_tile(Rymdboard* env, int x, int y, int shape[][2], int num_cells, int tile_type) {
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        int idx = by * BOARD_SIZE + bx;
        
        env->board_owner[idx] = 1;  // Player owns
        env->board_structure[idx] = tile_type;
    }
}

static int count_connected_crystals(Rymdboard* env) {
    int count = 0;
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        int cx = env->crystals[i].x;
        int cy = env->crystals[i].y;
        env->crystals[i].connected = 0;
        
        for (int d = 0; d < 4; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            
            if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) {
                continue;
            }
            
            int idx = ny * BOARD_SIZE + nx;
            
            if (env->board_owner[idx] == 1 &&
                env->board_structure[idx] == TILE_EXTRACTION_BUILDING) {
                env->crystals[i].connected = 1;
                count++;
                break;
            }
        }
    }
    
    return count;
}

static float get_min_distance_to_crystals(Rymdboard* env) {
    float min_dist = 20.0f;
    
    // Find all player tiles
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            if (env->board_owner[idx] != 1) continue;
            
            // Calculate distance to unconnected crystals
            for (int i = 0; i < NUM_CRYSTALS; i++) {
                if (env->crystals[i].connected) continue;
                
                int dx = abs(x - env->crystals[i].x);
                int dy = abs(y - env->crystals[i].y);
                float dist = (float)(dx + dy);
                
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
    }
    
    return min_dist;
}

static int decode_action(int action, int* card_idx, int* x, int* y, int* rotation) {
    // action = card_idx * 400 + x * 40 + y * 4 + rotation
    if (action < 0 || action >= NUM_ACTIONS) {
        return 0;
    }
    
    *rotation = action % 4;
    int rem = action / 4;
    *y = rem % 10;
    rem = rem / 10;
    *x = rem % 10;
    *card_idx = rem / 10;
    
    return 1;
}

// Observation format:
// - Board state: 100 values (one-hot would be too large, use integers)
// - Market: 7 * 30 = 210 values (shape as 5x5 + metadata)
// - Crystal info: 3 * 4 = 12 values (x, y, connected, amount)
// Total: ~322 floats

#define OBS_BOARD_SIZE 100
#define OBS_MARKET_SIZE 210  // 7 cards * 30 features each
#define OBS_CRYSTAL_SIZE 12
#define OBS_TOTAL_SIZE (OBS_BOARD_SIZE + OBS_MARKET_SIZE + OBS_CRYSTAL_SIZE)

static void fill_observation(Rymdboard* env) {
    int idx = 0;
    
    // Board observation: encode structure type (0-4) as float
    for (int i = 0; i < BOARD_TILES; i++) {
        // Encode: empty=0, street=1, extraction=2, base=3, crystal=4
        float val = 0.0f;
        if (env->board_feature[i] == TILE_CRYSTAL_NODE) {
            val = 4.0f;
        } else if (env->board_structure[i] != TILE_EMPTY) {
            val = (float)env->board_structure[i];
        }
        // Also encode ownership
        if (env->board_owner[i] == 1) {
            val += 10.0f;  // Add 10 if player owns
        }
        env->observations[idx++] = val / 14.0f;  // Normalize to 0-1
    }
    
    // Market cards
    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) 
            ? &env->building_market[c] 
            : &env->street_market[c - NUM_BUILDING_SLOTS];
        
        // Is building
        env->observations[idx++] = card->is_building ? 1.0f : 0.0f;
        
        // Points (normalized)
        env->observations[idx++] = card->static_points / 5.0f;
        
        // Num cells
        env->observations[idx++] = card->shape.num_cells / 5.0f;
        
        // Shape as 5x5 grid (25 values)
        float shape_grid[25] = {0};
        for (int i = 0; i < card->shape.num_cells; i++) {
            int sx = card->shape.cells[i][0];
            int sy = card->shape.cells[i][1];
            if (sx >= 0 && sx < 5 && sy >= 0 && sy < 5) {
                shape_grid[sy * 5 + sx] = 1.0f;
            }
        }
        for (int i = 0; i < 25; i++) {
            env->observations[idx++] = shape_grid[i];
        }
        
        // Padding to 30 features per card
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
    }
    
    // Crystal info
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        env->observations[idx++] = env->crystals[i].x / (float)(BOARD_SIZE - 1);
        env->observations[idx++] = env->crystals[i].y / (float)(BOARD_SIZE - 1);
        env->observations[idx++] = env->crystals[i].connected ? 1.0f : 0.0f;
        env->observations[idx++] = env->crystals[i].amount / 10.0f;
    }
}

void add_log(Rymdboard* env) {
    env->log.perf += (env->crystals_connected >= 2) ? 1.0f : 0.0f;
    env->log.score += (float)env->crystals_connected;
    env->log.episode_length += (float)env->tick;
    env->log.crystals_connected += (float)env->crystals_connected;
    env->log.valid_placements += (float)env->valid_placements;
    env->log.invalid_placements += (float)env->invalid_placements;
    env->log.n++;
}

void c_reset(Rymdboard* env) {
    // Clear board
    memset(env->board_owner, 0, BOARD_TILES);
    memset(env->board_structure, TILE_EMPTY, BOARD_TILES);
    memset(env->board_feature, TILE_EMPTY, BOARD_TILES);
    
    // Increment reset counter and update RNG state for variety
    env->reset_count++;
    env->rng_state = env->seed + env->reset_count * 1000003;
    if (env->rng_state == 0) env->rng_state = 1;  // xorshift needs non-zero
    
    // Reset stats
    env->tick = 0;
    env->valid_placements = 0;
    env->invalid_placements = 0;
    env->crystals_connected = 0;
    
    // Setup board
    place_start_base(env);
    place_crystal_nodes(env);
    init_decks(env);
    
    // Fill observation
    fill_observation(env);
}

void c_step(Rymdboard* env) {
    env->tick++;
    
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = REWARD_STEP_PENALTY;
    
    // Decode action
    int card_idx, x, y, rotation;
    if (!decode_action(action, &card_idx, &x, &y, &rotation)) {
        // Invalid action - terminate episode immediately
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(3.0f * (float)env->frameskip / 60.0f);
        }
        c_reset(env);
        return;
    }
    
    // Get card from market
    Card* card = NULL;
    int is_building = (card_idx < NUM_BUILDING_SLOTS);
    
    if (is_building) {
        card = &env->building_market[card_idx];
    } else {
        int street_idx = card_idx - NUM_BUILDING_SLOTS;
        if (street_idx < NUM_STREET_SLOTS) {
            card = &env->street_market[street_idx];
        }
    }
    
    if (card == NULL) {
        // Invalid card - terminate episode immediately
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(3.0f * (float)env->frameskip / 60.0f);
        }
        c_reset(env);
        return;
    }
    
    // Get distance before placement
    float prev_dist = get_min_distance_to_crystals(env);
    int prev_connected = count_connected_crystals(env);
    
    // Get rotated shape
    int rotated_shape[MAX_SHAPE_SIZE][2];
    int raw_shape[MAX_SHAPE_SIZE][2];
    for (int i = 0; i < card->shape.num_cells; i++) {
        raw_shape[i][0] = card->shape.cells[i][0];
        raw_shape[i][1] = card->shape.cells[i][1];
    }
    rotate_shape(raw_shape, card->shape.num_cells, rotation, rotated_shape);
    
    // Check if placement is valid (no overlap, in bounds)
    if (!can_place_tile(env, x, y, rotated_shape, card->shape.num_cells)) {
        // Invalid placement - terminate episode immediately
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(3.0f * (float)env->frameskip / 60.0f);
        }
        c_reset(env);
        return;
    }
    
    // Check if connected to network
    if (!is_connected_to_network(env, x, y, rotated_shape, card->shape.num_cells)) {
        // Not connected to network - terminate episode immediately
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(3.0f * (float)env->frameskip / 60.0f);
        }
        c_reset(env);
        return;
    }
    
    // Place the tile
    int tile_type = is_building ? TILE_EXTRACTION_BUILDING : TILE_STREET;
    place_tile(env, x, y, rotated_shape, card->shape.num_cells, tile_type);

    // Replace the used card with a new random one from the deck
    replace_used_card(env, card_idx);

    env->valid_placements++;
    env->rewards[0] += REWARD_VALID_PLACEMENT;
    
    // Distance reward
    float curr_dist = get_min_distance_to_crystals(env);
    if (curr_dist < prev_dist) {
        env->rewards[0] += REWARD_DISTANCE_SCALE * (prev_dist - curr_dist);
    }
    
    // Connection reward
    int curr_connected = count_connected_crystals(env);
    env->crystals_connected = curr_connected;
    
    if (curr_connected > prev_connected) {
        env->rewards[0] += REWARD_CONNECTION;
    }
    
    // Win condition: 2+ crystals connected
    if (curr_connected >= 2) {
        env->rewards[0] += REWARD_WIN_BONUS;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(3.0f * (float)env->frameskip / 60.0f);
        }
        c_reset(env);
        return;
    }
    
    // Truncation
    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            sleep(1);
        }
        c_reset(env);
        return;
    }
    
    env->log.episode_return += env->rewards[0];
    fill_observation(env);
}

// ============================================================================
// Helper: Encode action from (card_idx, x, y, rotation)
// ============================================================================

static int encode_action(int card_idx, int x, int y, int rotation) {
    // action = card_idx * 400 + x * 40 + y * 4 + rotation
    return card_idx * 400 + x * 40 + y * 4 + rotation;
}

// ============================================================================
// Rendering with Raylib
// ============================================================================

// Colors (PufferLib style)
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_WHITE = {241, 241, 241, 255};
static const Color PUFF_YELLOW = {255, 255, 0, 255};
static const Color PUFF_GREEN = {0, 187, 0, 255};
static const Color PUFF_PURPLE = {187, 0, 187, 255};
static const Color PUFF_DARK_CYAN = {0, 100, 100, 255};
static const Color PUFF_GRID = {20, 60, 60, 255};

#define CELL_SIZE 48
#define MARKET_WIDTH 200
#define INFO_HEIGHT 80
#define WINDOW_WIDTH (BOARD_SIZE * CELL_SIZE + MARKET_WIDTH)
#define WINDOW_HEIGHT (BOARD_SIZE * CELL_SIZE + INFO_HEIGHT)

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

void c_render(Rymdboard* env) {
    if (!IsWindowReady()) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Rymdboard - PufferLib Ocean");
        SetTargetFPS(60 / env->frameskip);
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    // Draw grid
    for (int i = 0; i <= BOARD_SIZE; i++) {
        // Vertical lines
        DrawLine(i * CELL_SIZE, 0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE, PUFF_GRID);
        // Horizontal lines
        DrawLine(0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE, i * CELL_SIZE, PUFF_GRID);
    }
    
    // Draw board cells
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            int px = x * CELL_SIZE;
            int py = y * CELL_SIZE;
            
            // Draw features first (crystal nodes)
            if (env->board_feature[idx] == TILE_CRYSTAL_NODE) {
                // Check if connected
                int connected = 0;
                for (int i = 0; i < NUM_CRYSTALS; i++) {
                    if (env->crystals[i].x == x && env->crystals[i].y == y) {
                        connected = env->crystals[i].connected;
                        break;
                    }
                }
                Color c = connected ? PUFF_GREEN : PUFF_YELLOW;
                DrawCircle(px + CELL_SIZE/2, py + CELL_SIZE/2, CELL_SIZE/3, c);
                DrawText("C", px + CELL_SIZE/2 - 5, py + CELL_SIZE/2 - 8, 16, PUFF_BACKGROUND);
            }
            
            // Draw structures
            if (env->board_structure[idx] != TILE_EMPTY) {
                Color c;
                const char* label = "";
                
                switch (env->board_structure[idx]) {
                    case TILE_STREET:
                        c = PUFF_CYAN;
                        label = "-";
                        break;
                    case TILE_EXTRACTION_BUILDING:
                        c = PUFF_PURPLE;
                        label = "E";
                        break;
                    case TILE_START_BASE:
                        c = PUFF_DARK_CYAN;
                        label = "B";
                        break;
                    default:
                        c = PUFF_WHITE;
                        label = "?";
                }
                
                DrawRectangle(px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4, c);
                DrawText(label, px + CELL_SIZE/2 - 4, py + CELL_SIZE/2 - 8, 16, PUFF_BACKGROUND);
            }
        }
    }
    
    // Draw coordinate labels
    for (int i = 0; i < BOARD_SIZE; i++) {
        char buf[4];
        snprintf(buf, sizeof(buf), "%d", i);
        // X axis at bottom
        DrawText(buf, i * CELL_SIZE + CELL_SIZE/2 - 4, BOARD_SIZE * CELL_SIZE + 2, 12, PUFF_WHITE);
    }
    
    // Draw market panel (right side)
    int market_x = BOARD_SIZE * CELL_SIZE + 10;
    int market_y = 10;
    
    DrawText("MARKET", market_x, market_y, 20, PUFF_WHITE);
    market_y += 30;
    
    // Buildings (0-3)
    DrawText("Buildings:", market_x, market_y, 14, PUFF_CYAN);
    market_y += 18;
    
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", i);
        DrawText(buf, market_x, market_y, 12, PUFF_WHITE);
        
        // Draw shape preview
        draw_shape_preview(market_x + 30, market_y - 5, &env->building_market[i], 0, PUFF_PURPLE, 12);
        market_y += 50;
    }
    
    // Streets (4-6)
    DrawText("Streets:", market_x, market_y, 14, PUFF_CYAN);
    market_y += 18;
    
    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", i + NUM_BUILDING_SLOTS);
        DrawText(buf, market_x, market_y, 12, PUFF_WHITE);
        
        // Draw shape preview
        draw_shape_preview(market_x + 30, market_y - 5, &env->street_market[i], 0, PUFF_CYAN, 12);
        market_y += 50;
    }
    
    // Draw info panel (bottom)
    int info_y = BOARD_SIZE * CELL_SIZE + 20;
    
    char info[128];
    snprintf(info, sizeof(info), "Step: %d  Connected: %d/3  Reward: %.2f", 
             env->tick, env->crystals_connected, env->rewards ? env->rewards[0] : 0.0f);
    DrawText(info, 10, info_y, 16, PUFF_WHITE);
    
    // Controls help
    DrawText("Enter: card x y rot (e.g. '4 3 3 0')", 10, info_y + 20, 12, PUFF_CYAN);
    DrawText("ESC: quit | R: reset", 10, info_y + 35, 12, PUFF_CYAN);
    
    DrawText("ESC: quit | R: reset", 10, info_y + 35, 12, PUFF_CYAN);
    
    // Draw last action
    int action = env->actions[0];
    int rot = action % 4;
    int rem = action / 4;
    int y = rem % 10;
    rem = rem / 10;
    int x = rem % 10;
    int card_idx = rem / 10;
    
    char action_buf[64];
    snprintf(action_buf, sizeof(action_buf), "Action: Card %d at (%d, %d) Rot %d", card_idx, x, y, rot);
    DrawText(action_buf, 10, info_y + 55, 16, PUFF_YELLOW);
    
    EndDrawing();
}

void c_close(Rymdboard* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

// ============================================================================
// Console helpers for interactive play
// ============================================================================

static void print_board(Rymdboard* env) {
    printf("\n   ");
    for (int x = 0; x < BOARD_SIZE; x++) {
        printf(" %d", x);
    }
    printf("\n   ");
    for (int x = 0; x < BOARD_SIZE; x++) {
        printf("--");
    }
    printf("\n");
    
    for (int y = 0; y < BOARD_SIZE; y++) {
        printf("%2d|", y);
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            char c = '.';
            
            if (env->board_feature[idx] == TILE_CRYSTAL_NODE) {
                // Check if connected
                int connected = 0;
                for (int i = 0; i < NUM_CRYSTALS; i++) {
                    if (env->crystals[i].x == x && env->crystals[i].y == y) {
                        connected = env->crystals[i].connected;
                        break;
                    }
                }
                c = connected ? '*' : 'C';
            } else if (env->board_structure[idx] == TILE_START_BASE) {
                c = 'B';
            } else if (env->board_structure[idx] == TILE_STREET) {
                c = '-';
            } else if (env->board_structure[idx] == TILE_EXTRACTION_BUILDING) {
                c = 'E';
            }
            
            printf(" %c", c);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_market(Rymdboard* env) {
    printf("=== MARKET ===\n");
    printf("Buildings (0-3):\n");
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        Card* card = &env->building_market[i];
        printf("  [%d] %d cells, %d pts, shape: ", i, card->shape.num_cells, card->static_points);
        for (int j = 0; j < card->shape.num_cells; j++) {
            printf("(%d,%d)", card->shape.cells[j][0], card->shape.cells[j][1]);
        }
        printf("\n");
    }
    printf("Streets (4-6):\n");
    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        Card* card = &env->street_market[i];
        printf("  [%d] %d cells, %d pts, shape: ", i + NUM_BUILDING_SLOTS, card->shape.num_cells, card->static_points);
        for (int j = 0; j < card->shape.num_cells; j++) {
            printf("(%d,%d)", card->shape.cells[j][0], card->shape.cells[j][1]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_crystals(Rymdboard* env) {
    printf("=== CRYSTALS (%d connected) ===\n", env->crystals_connected);
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        printf("  Crystal %d: (%d, %d) - %s\n", 
               i, env->crystals[i].x, env->crystals[i].y,
               env->crystals[i].connected ? "CONNECTED" : "not connected");
    }
    printf("\n");
}

static void print_help(void) {
    printf("=== COMMANDS ===\n");
    printf("  <card> <x> <y> <rot>  - Place card (e.g. '4 3 3 0')\n");
    printf("  r                     - Reset game\n");
    printf("  q                     - Quit\n");
    printf("  h                     - Show this help\n");
    printf("\nCards: 0-3 = buildings (extraction), 4-6 = streets\n");
    printf("Rotations: 0-3 (90 degree increments)\n\n");
}

