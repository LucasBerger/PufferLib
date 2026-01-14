/* Crystalboard: A tile-placement game environment for PufferLib Ocean
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


// Removed fixed board dimensions
// #define BOARD_SIZE 10 -- Now dynamic
// #define BOARD_TILES (BOARD_SIZE * BOARD_SIZE) -- Now dynamic

// Market sizes (still fixed)
#define NUM_BUILDING_SLOTS 4
#define NUM_STREET_SLOTS 3
#define TOTAL_MARKET_SLOTS 7

// Action space dependent on grid size: 7 cards * W * H * 4 rotations
// #define NUM_ACTIONS 2800 -- Now dynamic

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
    
    // Dynamic Grid Dimensions
    int width;
    int height; // We'll assume square for now usually, but good to have both
    int board_tiles; // width * height
    
    // Game state (Dynamic arrays)
    unsigned char* board_owner;      // 0=empty, 1=player
    unsigned char* board_structure;  // Tile type
    unsigned char* board_feature;    // Features like crystal nodes
    
    // Market state - cards are drawn from decks and replaced when used
    Card building_market[NUM_BUILDING_SLOTS];
    Card street_market[NUM_STREET_SLOTS];
    // Available cards in decks (shapes that can be drawn)
    int building_deck[21];  // 3 copies of each of 7 shapes = 21 cards
    int street_deck[21];    // Same for streets
    int building_deck_size;
    int street_deck_size;
    int last_total_distance;
    
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
} Crystalboard;

// Reward constants (from research/reward_constants.py)
// Invalid moves now terminate the episode, so penalty doesn't need to be extreme
#define REWARD_STEP_PENALTY -0.02f
#define REWARD_INVALID_MOVE -0.02f
#define REWARD_VALID_PLACEMENT 0.0f
#define REWARD_DISTANCE_SCALE 0.05f
#define REWARD_CONNECTION 8.0f
#define REWARD_WIN_BONUS 10.0f
#define REWARD_NO_VALID_ACTIONS -10.0f

// Invalid moves now terminate the episode, so penalty doesn't need to be extreme
// #define REWARD_STEP_PENALTY -0.1f
// #define REWARD_INVALID_MOVE -0.1f
// #define REWARD_VALID_PLACEMENT 0.0f
// #define REWARD_DISTANCE_SCALE 0.2f
// #define REWARD_CONNECTION 5.0f
// #define REWARD_WIN_BONUS 10.0f
// #define REWARD_NO_VALID_ACTIONS -1.0f
// #define REWARD_NO_VALID_ACTIONS -1.0f
#define REWARD_STREET_BLOCKING_PENALTY -0.5f
#define REWARD_FAR_BUILDING_PENALTY -0.0f

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
void add_log(Crystalboard* env);
void c_reset(Crystalboard* env);
void c_step(Crystalboard* env);
void c_render(Crystalboard* env);
void c_draw_internal(Crystalboard* env);
void c_close(Crystalboard* env);

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
static void init_card_decks(Crystalboard* env) {
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

static void replace_used_card(Crystalboard* env, int card_idx) {
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

static void init_decks(Crystalboard* env) {
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

static void place_start_base(Crystalboard* env) {
    // Place 2x2 start base always at (1,1) if possible
    int x = 1, y = 1;
    for (int i = 0; i < SHAPE_BASE_SIZE; i++) {
        int bx = x + SHAPE_BASE[i][0];
        int by = y + SHAPE_BASE[i][1];
        if (bx < env->width && by < env->height) {
            int idx = by * env->width + bx;
            env->board_owner[idx] = 1;  // Player owns it
            env->board_structure[idx] = TILE_START_BASE;
        }
    }
}

static void place_crystal_nodes(Crystalboard* env) {
    // Place crystal nodes in 3 quadrants (excluding top-left where base is)
    // We scale the quadrants based on env->width/height
    
    int half_w = env->width / 2;
    int half_h = env->height / 2;
    
    // Top-right: x=[half_w, w-1], y=[0, half_h-1]
    env->crystals[0].x = half_w + (xorshift32(&env->rng_state) % half_w);
    env->crystals[0].y = xorshift32(&env->rng_state) % half_h;
    env->crystals[0].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[0].connected = 0;
    
    // Bottom-left: x=[0, half_w-1], y=[half_h, h-1]
    env->crystals[1].x = xorshift32(&env->rng_state) % half_w;
    env->crystals[1].y = half_h + (xorshift32(&env->rng_state) % half_h);
    env->crystals[1].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[1].connected = 0;
    
    // Bottom-right: x=[half_w, w-1], y=[half_h, h-1]
    env->crystals[2].x = half_w + (xorshift32(&env->rng_state) % half_w);
    env->crystals[2].y = half_h + (xorshift32(&env->rng_state) % half_h);
    env->crystals[2].amount = 5 + (xorshift32(&env->rng_state) % 2);
    env->crystals[2].connected = 0;
    
    // Place on board feature grid
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        int idx = env->crystals[i].y * env->width + env->crystals[i].x;
        if (idx < env->board_tiles) {
             env->board_feature[idx] = TILE_CRYSTAL_NODE;
        }
    }
}

static void rotate_shape(const int shape[][2], int num_cells, int rotation, int out_shape[][2]) {
    // Rotate shape 90 degrees clockwise 'rotation' times
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

static int can_place_tile(Crystalboard* env, int x, int y, int shape[][2], int num_cells) {
    // Check bounds and overlap
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        
        // Check bounds
        if (bx < 0 || bx >= env->width || by < 0 || by >= env->height) {
            return 0;
        }
        
        int idx = by * env->width + bx;
        
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

static int is_connected_to_network(Crystalboard* env, int x, int y, int shape[][2], int num_cells) {
    // Check if any cell of the shape is adjacent to player's street or base
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        
        for (int d = 0; d < 4; d++) {
            int nx = bx + dx[d];
            int ny = by + dy[d];
            
            if (nx < 0 || nx >= env->width || ny < 0 || ny >= env->height) {
                continue;
            }
            
            int idx = ny * env->width + nx;
            
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

static void place_tile(Crystalboard* env, int x, int y, int shape[][2], int num_cells, int tile_type) {
    for (int i = 0; i < num_cells; i++) {
        int bx = x + shape[i][0];
        int by = y + shape[i][1];
        int idx = by * env->width + bx;
        
        env->board_owner[idx] = 1;  // Player owns
        env->board_structure[idx] = tile_type;
    }
}

static int count_connected_crystals(Crystalboard* env) {
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
            
            if (nx < 0 || nx >= env->width || ny < 0 || ny >= env->height) {
                continue;
            }
            
            int idx = ny * env->width + nx;
            
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

static float get_min_distance_to_crystals(Crystalboard* env) {
    float min_dist = (float)(env->width + env->height);
    
    // Find all player tiles
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
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

static int decode_action(Crystalboard* env, int action, int* card_idx, int* x, int* y, int* rotation) {
    // action definition depends on width/height
    // We'll flatten strictly:
    // action = card_idx * (W*H*4) + (y * W + x) * 4 + rotation (Standard PufferLib spatial style usually y*W+x)
    
    int num_pos_rot = env->width * env->height * 4;
    int total_actions = TOTAL_MARKET_SLOTS * num_pos_rot;
    
    if (action < 0 || action >= total_actions) {
        return 0;
    }
    
    *rotation = action % 4;
    int rem = action / 4;
    
    int pos_idx = rem % (env->width * env->height);
    *x = pos_idx % env->width;
    *y = pos_idx / env->width;
    
    *card_idx = rem / (env->width * env->height);
    
    return 1;
}

// Helper to encode action (mostly for interactive play)
static int encode_action(Crystalboard* env, int card_idx, int x, int y, int rotation) {
    int pos_idx = y * env->width + x;
    return card_idx * (env->width * env->height * 4) + pos_idx * 4 + rotation;
}


#define OBS_MARKET_SIZE 210  // 7 cards * 30 features each
#define OBS_CRYSTAL_SIZE 12
// OBS_BOARD_SIZE is dynamic: env->board_tiles

// Total Obs Size matches what python expects?
// Binding/Python side handles the buffer size allocation.
// We just write to current `env->observations`.
// OBS_REAL_SIZE = board_tiles + 210 + 12
// OBS_TOTAL_SIZE = OBS_REAL_SIZE + NUM_ACTIONS (for action mask)

static int fill_observation(Crystalboard* env) {
    int idx = 0;
    
    // Board observation
    for (int i = 0; i < env->board_tiles; i++) {
        float val = 0.0f;
        if (env->board_feature[i] == TILE_CRYSTAL_NODE) {
            val = 4.0f;
        } else if (env->board_structure[i] != TILE_EMPTY) {
            val = (float)env->board_structure[i];
        }
        if (env->board_owner[i] == 1) {
            val += 10.0f;
        }
        env->observations[idx++] = val / 44.0f;
    }
    
    // Market cards
    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) 
            ? &env->building_market[c] 
            : &env->street_market[c - NUM_BUILDING_SLOTS];
        
        env->observations[idx++] = card->is_building ? 1.0f : 0.0f;
        env->observations[idx++] = card->static_points / 5.0f;
        env->observations[idx++] = card->shape.num_cells / 5.0f;
        
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
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
    }
    
    // Crystal info
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        env->observations[idx++] = env->crystals[i].x / (float)(env->width - 1);
        env->observations[idx++] = env->crystals[i].y / (float)(env->height - 1);
        env->observations[idx++] = env->crystals[i].connected ? 1.0f : 0.0f;
        env->observations[idx++] = env->crystals[i].amount / 10.0f;
    }

    // Action mask
    // Values start at idx
    // Total actions = 7 * W * H * 4
    
    int rotated_shapes[TOTAL_MARKET_SLOTS][4][MAX_SHAPE_SIZE][2];
    int num_cells[TOTAL_MARKET_SLOTS];

    for (int c = 0; c < TOTAL_MARKET_SLOTS; c++) {
        Card* card = (c < NUM_BUILDING_SLOTS) 
            ? &env->building_market[c] 
            : &env->street_market[c - NUM_BUILDING_SLOTS];
        
        num_cells[c] = card->shape.num_cells;
        int raw_shape[MAX_SHAPE_SIZE][2];
        for (int i = 0; i < card->shape.num_cells; i++) {
            raw_shape[i][0] = card->shape.cells[i][0];
            raw_shape[i][1] = card->shape.cells[i][1];
        }
        
        for (int r = 0; r < 4; r++) {
            rotate_shape(raw_shape, card->shape.num_cells, r, rotated_shapes[c][r]);
        }
    }

    int num_actions = TOTAL_MARKET_SLOTS * env->width * env->height * 4;
    // We must assume the observations buffer is large enough!
    // env->observations allocated size should ideally be passed or known.
    // For now we trust Python side calculated it correctly.
    
    for (int i = 0; i < num_actions; i++) {
        env->observations[idx + i] = 0.0f;
    }

    int valid_action_count = 0;

    // This loop is expensive for large grids (30x30x7x4 = ~25k iterations)
    for (int card_idx = 0; card_idx < TOTAL_MARKET_SLOTS; card_idx++) {
        for (int x = 0; x < env->width; x++) {
            for (int y = 0; y < env->height; y++) {
                for (int r = 0; r < 4; r++) {
                    if (can_place_tile(env, x, y, rotated_shapes[card_idx][r], num_cells[card_idx]) &&
                        is_connected_to_network(env, x, y, rotated_shapes[card_idx][r], num_cells[card_idx])) {
                        
                        int action_idx = encode_action(env, card_idx, x, y, r);
                        env->observations[idx + action_idx] = 1.0f;
                        valid_action_count++;
                    }
                }
            }
        }
    }
    
    return valid_action_count;
}

void add_log(Crystalboard* env) {
    env->log.perf += (env->crystals_connected >= 3) ? 1.0f : 0.0f;
    env->log.score += (float)env->crystals_connected;
    env->log.episode_length += (float)env->tick;
    env->log.crystals_connected += (float)env->crystals_connected;
    env->log.valid_placements += (float)env->valid_placements;
    env->log.invalid_placements += (float)env->invalid_placements;
    env->log.n++;
}

// Rendering with Raylib (Dynamic)
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_WHITE = {241, 241, 241, 255};
static const Color PUFF_YELLOW = {255, 255, 0, 255};
static const Color PUFF_GREEN = {0, 187, 0, 255};
static const Color PUFF_PURPLE = {187, 0, 187, 255};
static const Color PUFF_DARK_CYAN = {0, 100, 100, 255};
static const Color PUFF_GRID = {20, 60, 60, 255};
static const Color PUFF_GRAY = {130, 130, 130, 255};

// Forward declaration of internal drawing to keep it separable if needed
void c_draw_internal(Crystalboard* env);

void c_reset(Crystalboard* env) {
    // Check if board memory is allocated
    if (!env->board_owner || !env->board_structure || !env->board_feature) {
        // If not allocated (e.g. standalone mode might fail here if not careful), return or error.
        // Standalone must allocate before calling c_reset.
        return;
    }
    
    // Clear board
    memset(env->board_owner, 0, env->board_tiles);
    memset(env->board_structure, TILE_EMPTY, env->board_tiles);
    memset(env->board_feature, TILE_EMPTY, env->board_tiles);
    env->last_total_distance = 0;
    
    // Increment reset counter
    env->reset_count++;
    env->rng_state = env->seed + env->reset_count * 1000003;
    if (env->rng_state == 0) env->rng_state = 1;
    
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

// ... (previous code)

void c_render(Crystalboard* env) {
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim; // Reduced from 1000 to 800 for smaller screens
    if (cell_size > 48) cell_size = 48; // Cap size
    if (cell_size < 8) cell_size = 8; // Min size
    
    int market_width = 200;
    int win_w = env->width * cell_size + market_width;
    int win_h = env->height * cell_size + 80;

    if (!IsWindowReady()) {
        InitWindow(win_w, win_h, "Crystalboard - PufferLib Ocean");
        SetTargetFPS(60);
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        // Just return, let caller handle exit
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    c_draw_internal(env);
    EndDrawing();
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

void c_draw_internal(Crystalboard* env) {
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim; // Same scaling as render
    if (cell_size > 48) cell_size = 48;
    if (cell_size < 8) cell_size = 8;
    
    // Draw grid
    for (int i = 0; i <= env->width; i++) {
        DrawLine(i * cell_size, 0, i * cell_size, env->height * cell_size, PUFF_GRID);
    }
    for (int i = 0; i <= env->height; i++) {
        DrawLine(0, i * cell_size, env->width * cell_size, i * cell_size, PUFF_GRID);
    }
    
    // Draw board cells
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            int px = x * cell_size;
            int py = y * cell_size;
            
            // Draw features first (crystal nodes)
            if (env->board_feature[idx] == TILE_CRYSTAL_NODE) {
                int connected = 0;
                for (int i = 0; i < NUM_CRYSTALS; i++) {
                    if (env->crystals[i].x == x && env->crystals[i].y == y) {
                        connected = env->crystals[i].connected;
                        break;
                    }
                }
                Color c = connected ? PUFF_GREEN : PUFF_YELLOW;
                DrawCircle(px + cell_size/2, py + cell_size/2, cell_size/3, c);
                if (cell_size >= 20) DrawText("C", px + cell_size/2 - 5, py + cell_size/2 - 8, cell_size/3, PUFF_BACKGROUND);
            }
            
            // Draw structures
            if (env->board_structure[idx] != TILE_EMPTY) {
                Color c;
                switch (env->board_structure[idx]) {
                    case TILE_STREET: c = PUFF_CYAN; break;
                    case TILE_EXTRACTION_BUILDING: c = PUFF_PURPLE; break;
                    case TILE_START_BASE: c = PUFF_DARK_CYAN; break;
                    default: c = PUFF_WHITE;
                }
                DrawRectangle(px + 1, py + 1, cell_size - 2, cell_size - 2, c);
            }
        }
    }
    
    // Draw market panel
    int market_x = env->width * cell_size + 10;
    int market_y = 10;
    DrawText("MARKET", market_x, market_y, 20, PUFF_WHITE);
    market_y += 30;
    
// ... (previous code)

    // Buildings
    for (int i = 0; i < NUM_BUILDING_SLOTS; i++) {
        char num[4];
        snprintf(num, sizeof(num), "%d", i);
        DrawText(num, market_x, market_y + 10, 20, PUFF_WHITE);
        draw_shape_preview(market_x + 30, market_y, &env->building_market[i], 0, PUFF_PURPLE, 12);
        market_y += 50;
    }
    
    market_y += 10;
    // Streets
    for (int i = 0; i < NUM_STREET_SLOTS; i++) {
        char num[4];
        snprintf(num, sizeof(num), "%d", i + NUM_BUILDING_SLOTS);
        DrawText(num, market_x, market_y + 10, 20, PUFF_WHITE);
        draw_shape_preview(market_x + 30, market_y, &env->street_market[i], 0, PUFF_CYAN, 12);
        market_y += 50;
    }
    
    // -- RESTORED TEXT --
    int text_y = env->height * cell_size + 10;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "Steps: %d/%d  |  Crystals: %d/3  |  Return: %.1f", 
             env->tick, env->max_steps, env->crystals_connected, env->log.episode_return);
    DrawText(buffer, 10, text_y, 20, PUFF_WHITE);
    
    DrawText("Controls: 0-6 Select Card | WASD Move | Q/E Rotate | Space Place | R Reset", 10, text_y + 25, 10, PUFF_GRAY);
}

void c_step(Crystalboard* env) {
    env->tick++;
    
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = REWARD_STEP_PENALTY;
    
    int card_idx, x, y, rotation;
    if (!decode_action(env, action, &card_idx, &x, &y, &rotation)) {
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
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
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
    // Get distance before placement
    float prev_dist = get_min_distance_to_crystals(env);
    int prev_connected = count_connected_crystals(env);
    
    int rotated_shape[MAX_SHAPE_SIZE][2];
    int raw_shape[MAX_SHAPE_SIZE][2];
    for (int i = 0; i < card->shape.num_cells; i++) {
        raw_shape[i][0] = card->shape.cells[i][0];
        raw_shape[i][1] = card->shape.cells[i][1];
    }
    rotate_shape(raw_shape, card->shape.num_cells, rotation, rotated_shape);
    
    if (!can_place_tile(env, x, y, rotated_shape, card->shape.num_cells)) {
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
             c_render(env);
             WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
    if (!is_connected_to_network(env, x, y, rotated_shape, card->shape.num_cells)) {
        env->rewards[0] = REWARD_INVALID_MOVE;
        env->invalid_placements++;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
             c_render(env);
             WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
    int tile_type = is_building ? TILE_EXTRACTION_BUILDING : TILE_STREET;
    place_tile(env, x, y, rotated_shape, card->shape.num_cells, tile_type);

    replace_used_card(env, card_idx);

    env->valid_placements++;
    env->rewards[0] += REWARD_VALID_PLACEMENT;

    // Penalty for blocking crystals with streets
    if (!is_building) {
        for (int i = 0; i < card->shape.num_cells; i++) {
            int bx = x + rotated_shape[i][0];
            int by = y + rotated_shape[i][1];
            
            int blocking = 0;
            for (int c = 0; c < NUM_CRYSTALS; c++) {
                 if (env->crystals[c].connected) continue;

                 int dist = abs(bx - env->crystals[c].x) + abs(by - env->crystals[c].y);
                 if (dist < 2) {
                     // Directly adjacent to an unconnected crystal with a street -> Blocking!
                     blocking = 1;
                     break;
                 }
            }
            if (blocking) {
                env->rewards[0] += REWARD_STREET_BLOCKING_PENALTY;
            }
        }
    }
    
    float current_total_dist = 0;
    int active_crystals = 0;
    for (int i = 0; i < NUM_CRYSTALS; i++) {
        if (!env->crystals[i].connected) {
            float min_dist_to_this_crystal = (float)(env->width + env->height);
            for (int idx = 0; idx < env->board_tiles; idx++) {
                if (env->board_owner[idx] == 1) {
                    int px = idx % env->width;
                    int py = idx / env->width;
                    int d = abs(px - env->crystals[i].x) + abs(py - env->crystals[i].y);
                    if (d < min_dist_to_this_crystal) min_dist_to_this_crystal = (float)d;
                }
            }
            current_total_dist += min_dist_to_this_crystal;
            active_crystals++;
        }
    }

    if (active_crystals > 0) {
        float distance_improvement = env->last_total_distance - current_total_dist;
        if (distance_improvement > 0) {
            env->rewards[0] += (distance_improvement * REWARD_DISTANCE_SCALE);
        }
    }
    env->last_total_distance = current_total_dist;
    
    int curr_connected = count_connected_crystals(env);
    env->crystals_connected = curr_connected;
    
    if (curr_connected > prev_connected) {
        env->rewards[0] += REWARD_CONNECTION;
    }

    // Penalty for placing extraction buildings too far from crystals
    if (is_building) {
        int min_dist_to_any_crystal = env->width + env->height;
        
        for (int i = 0; i < card->shape.num_cells; i++) {
            int bx = x + rotated_shape[i][0];
            int by = y + rotated_shape[i][1];
            
            for (int c = 0; c < NUM_CRYSTALS; c++) {
                int dist = abs(bx - env->crystals[c].x) + abs(by - env->crystals[c].y);
                if (dist < min_dist_to_any_crystal) {
                    min_dist_to_any_crystal = dist;
                }
            }
        }
        
        if (min_dist_to_any_crystal > 5) {
            env->rewards[0] += REWARD_FAR_BUILDING_PENALTY;
        }
    }
    
    if (curr_connected >= 3) {
        env->rewards[0] += REWARD_WIN_BONUS;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
            c_render(env);
            WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        if (env->render_mode) {
             c_render(env);
             WaitTime(4.0f);
        }
        c_reset(env);
        return;
    }
    
    int valid_actions = fill_observation(env);
    
    if (valid_actions == 0) {
        env->rewards[0] += REWARD_NO_VALID_ACTIONS;
        env->terminals[0] = 1;
    }

    env->log.episode_return += env->rewards[0];
    
    if (env->terminals[0]) {
        add_log(env);
        if (env->render_mode) {
             c_render(env);
             WaitTime(4.0f);
        }
        c_reset(env);
    }
}

void c_close(Crystalboard* env) {
    if (env->board_owner) free(env->board_owner);
    if (env->board_structure) free(env->board_structure);
    if (env->board_feature) free(env->board_feature);
    
    if (IsWindowReady()) {
        CloseWindow();
    }
}

// ============================================================================
// Console helpers for interactive play
// ============================================================================

static void print_board(Crystalboard* env) {
    printf("\n   ");
    for (int x = 0; x < env->width; x++) {
        printf(" %d", x % 10);
    }
    printf("\n   ");
    for (int x = 0; x < env->width; x++) {
        printf("--");
    }
    printf("\n");
    
    for (int y = 0; y < env->height; y++) {
        printf("%2d|", y);
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            char c = '.';
            
            if (env->board_feature[idx] == TILE_CRYSTAL_NODE) {
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

static void print_market(Crystalboard* env) {
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

static void print_crystals(Crystalboard* env) {
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
