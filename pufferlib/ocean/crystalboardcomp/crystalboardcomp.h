/* Crystalboard Competitive: A 4-player competitive environment
 * 
 * The goal is to extract resources from crystals.
 * 4 players start in corners.
 * Shared view centered on current player.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "raylib.h"

// Game Constants
#define NUM_PLAYERS 4

// Market sizes (same as original)
#define NUM_BUILDING_SLOTS 4
#define NUM_STREET_SLOTS 3
#define TOTAL_MARKET_SLOTS 7

// Crystal nodes: 4 starters + 4 sides + 2 center = 10
#define NUM_CRYSTALS 10

// Max shapes
#define MAX_SHAPE_SIZE 5
#define MAX_CRYSTAL_RESOURCES 10

// Tile types
#define TILE_EMPTY 0
#define TILE_STREET 1
#define TILE_EXTRACTION_BUILDING 2
#define TILE_START_BASE 3
#define TILE_CRYSTAL_NODE 4

// Shape definitions (Same as original)
typedef struct {
    int cells[MAX_SHAPE_SIZE][2];
    int num_cells;
} Shape;

typedef struct {
    Shape shape;
    int is_building;
    int static_points;
} Card;

typedef struct {
    int x;
    int y;
    int resources; // Remaining resources (starts at 10)
    int connections[NUM_PLAYERS]; // Number of connected buildings per player
} CrystalNode;

// Log struct (Standard PufferLib)
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float crystals_connected; // Average connected crystals per player? Or sum?
    float valid_placements;
    float invalid_placements;
    
    // Per-player return
    float p1_return;
    float p2_return;
    float p3_return;
    float p4_return;
    
    // Per-player stats
    float p1_crystal_collected;
    float p2_crystal_collected;
    float p3_crystal_collected;
    float p4_crystal_collected;
    
    float p1_crystals_connected;
    float p2_crystals_connected;
    float p3_crystals_connected;
    float p4_crystals_connected;
    
    float p1_win_rate;
    float p2_win_rate;
    float p3_win_rate;
    float p4_win_rate;
    
    float n;

    // Human Input (Exposed to Python)
    float human_action_ready;
    float last_human_action;
    float current_player;
} Log;

// Main environment struct
typedef struct {
    Log log;
    
    // Buffers
    float* observations;
    int* actions;
    float* rewards; // length 1 (shared?) or NUM_PLAYERS?
                    // PufferLib vectorization usually expects rewards[0] for the stepper.
                    // If we step one player, we return that player's reward.
    unsigned char* terminals;
    
    // Dimensions
    int width;
    int height;
    int board_tiles;
    
    // Game state
    unsigned char* board_owner;      // 0=empty, 1..4=player ID
    unsigned char* board_structure;  // Tile type
    unsigned char* board_feature;    // Features
    
    // Market
    Card building_market[NUM_BUILDING_SLOTS];
    Card street_market[NUM_STREET_SLOTS];
    int building_deck[21];
    int street_deck[21];
    int building_deck_size;
    int street_deck_size;
    
    // Crystals
    CrystalNode crystals[NUM_CRYSTALS];
    
    // Player State
    int current_player; // 0 to 3
    int scores[NUM_PLAYERS]; // Resources collected
    
    // Episode state
    int tick;         // Total moves (all players)
    int round;        // Increments every 4 ticks
    int max_steps;
    int seed;
    unsigned int rng_state;
    unsigned int reset_count;
    
    int consecutive_passes; // Track deadlocks
    
    // Stats
    int valid_placements; // For current episode
    int invalid_placements;
    
    int frameskip;
    int render_mode;
    int render_fps; // Target FPS for human render mode
    
    // Stats Accumulators (Temp storage for current episode)
    float acc_p1_crystal_collected;
    float acc_p2_crystal_collected;
    float acc_p3_crystal_collected;
    float acc_p4_crystal_collected;
    
    // Connection Accumulators (Sum over steps for average)
    float acc_p1_crystals_connected_sum;
    float acc_p2_crystals_connected_sum;
    float acc_p3_crystals_connected_sum;
    float acc_p4_crystals_connected_sum;
    
    // Reward Tracking
    float last_total_distance[NUM_PLAYERS];
    float current_episode_return;
    float current_p1_return;
    float current_p2_return;
    float current_p3_return;
    float current_p4_return;

    // Human Input State
    int cursor_x;
    int cursor_y;
    int selected_card;
    int rotation;
    int last_human_action;
    int human_action_ready;
} CrystalboardComp;

// Constants
#define REWARD_STEP_PENALTY -0.1f
#define REWARD_INVALID_MOVE -0.05f
#define REWARD_VALID_PLACEMENT 0.0f
#define REWARD_DISTANCE_SCALE 0.05f
#define REWARD_CONNECTION 8.0f
static const float REWARD_WIN_BONUS[] = {10.0f, 5.0f, 2.5f, 1.0f};
#define REWARD_STREET_BLOCKING_PENALTY -0.5f
#define REWARD_USELESS_STREET_PENALTY -0.1f // New penalty for non-distance-reducing streets
// Resource reward is 1.0 per resource?
#define REWARD_RESOURCE 0.0f

// Function Declarations
void c_reset(CrystalboardComp* env);
void c_step(CrystalboardComp* env);
void c_render(CrystalboardComp* env);
void c_close(CrystalboardComp* env);
