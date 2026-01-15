/* Quarryboard: A 4-player competitive environment with scarcity mechanics
 * 
 * The goal is to extract crystals.
 * 4 players start in corners.
 * Shared view centered on current player.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "raylib.h"

#define PUFF_WHITE (Color){255, 255, 255, 255}
#define PUFF_BLACK (Color){0, 0, 0, 255}
#define PUFF_RED (Color){230, 41, 55, 255}
#define PUFF_GREEN (Color){0, 228, 48, 255}
#define PUFF_BLUE (Color){0, 121, 241, 255}
#define PUFF_PURPLE (Color){200, 122, 255, 255}
#define PUFF_CYAN (Color){0, 228, 255, 255}
#define PUFF_YELLOW (Color){255, 255, 0, 255}
#define PUFF_GRID (Color){20, 60, 60, 255}
#define PUFF_BACKGROUND (Color){20, 20, 20, 255}
#ifndef RAYLIB_H
#define RAYLIB_H
typedef struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} Color;
typedef struct Rectangle {
    float x;
    float y;
    float width;
    float height;
} Rectangle;



// Mock Functions
static inline void InitWindow(int width, int height, const char *title) {}
static inline void CloseWindow(void) {}
static inline int WindowShouldClose(void) { return 1; } // Always close if no raylib
static inline void BeginDrawing(void) {}
static inline void EndDrawing(void) {}
static inline void ClearBackground(Color color) {}
static inline void DrawRectangle(int posX, int posY, int width, int height, Color color) {}
static inline void DrawRectangleLines(int posX, int posY, int width, int height, Color color) {}
static inline void DrawRectangleLinesEx(Rectangle rec, float lineThick, Color color) {}
static inline void DrawLine(int startPosX, int startPosY, int endPosX, int endPosY, Color color) {}
static inline void DrawCircle(int centerX, int centerY, float radius, Color color) {}
static inline void DrawText(const char *text, int posX, int posY, int fontSize, Color color) {}
static inline int MeasureText(const char *text, int fontSize) { return 0; }
static inline void SetTargetFPS(int fps) {}
static inline int IsWindowReady(void) { return 0; }
static inline int IsKeyPressed(int key) { return 0; }
static inline void PollInputEvents(void) {}

// Keys
#define KEY_R 82
#define KEY_ZERO 48
#define KEY_KP_0 320
#define KEY_LEFT 263
#define KEY_RIGHT 262
#define KEY_UP 265
#define KEY_DOWN 264
#define KEY_A 65
#define KEY_D 68
#define KEY_W 87
#define KEY_S 83
#define KEY_Q 81
#define KEY_E 69
#define KEY_SPACE 32
#define KEY_ENTER 257
#define KEY_LEFT_BRACKET 91
#define KEY_RIGHT_BRACKET 93
#define KEY_P 80
#define KEY_F 70

static inline const char* TextFormat(const char *text, ...) { return ""; }
#endif

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
#define TILE_QUARRY 5

// Shape definitions (Same as original)
typedef struct {
    int cells[MAX_SHAPE_SIZE][2];
    int num_cells;
} Shape;

typedef struct {
    Shape shape;
    int is_building;
    int is_quarry; // New: 1 if quarry, 0 if extraction (only valid if is_building=1)
    int static_points; // Still used? "crystal are now... tender... but extracted crystals are points"
                       // Maybe keep for compatibility or aux reward, but main objective changed.
} Card;

typedef struct {
    int x;
    int y;
    int resources; // Remaining resources (starts at 10)
    int connections[NUM_PLAYERS]; // Number of connected buildings per player
} CrystalNode;

// Quarry building tracker (per player)
#define MAX_QUARRY_BUILDINGS_PER_PLAYER 25
typedef struct {
    int x; // Anchor position (first cell placed)
    int y;
} QuarryBuilding;

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
    
    float p1_stone_collected;
    float p2_stone_collected;
    float p3_stone_collected;
    float p4_stone_collected;

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
    float* rewards; 
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
    
    // Quarry Buildings (per player)
    QuarryBuilding quarry_buildings[NUM_PLAYERS][MAX_QUARRY_BUILDINGS_PER_PLAYER];
    int quarry_count[NUM_PLAYERS];
    
    // Player State
    int current_player; // 0 to 3
    int scores[NUM_PLAYERS]; // Spendable Crystals (Current Inventory)
    int stones[NUM_PLAYERS]; // Spendable Stones (Current Inventory)
    int cumulative_crystals[NUM_PLAYERS]; // Total Extracted (Score for Winning)
    int finished[NUM_PLAYERS]; // 0=active, 1=finished
    int dead[NUM_PLAYERS]; // 0=alive, 1=dead (no possible moves)

    // Start Resources Config
    int start_crystals;
    int start_stones;

    // Episode state
    int tick;         // Total moves (all players)
    int active_turns; // Turns by active (non-finished, non-dead) players
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

    float acc_p1_stone_collected;
    float acc_p2_stone_collected;
    float acc_p3_stone_collected;
    float acc_p4_stone_collected;
    
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
} Quarryboard;

// Constants
#define REWARD_STEP_PENALTY -0.1f
#define REWARD_INVALID_MOVE -0.05f
#define REWARD_VALID_PLACEMENT 0.0f
#define REWARD_DISTANCE_SCALE 0.05f
#define REWARD_CONNECTION 8.0f
static const float REWARD_WIN_BONUS[] = {10.0f, 5.0f, 2.5f, 1.0f};
#define REWARD_STREET_BLOCKING_PENALTY -0.5f
#define REWARD_USELESS_STREET_PENALTY -0.1f 
#define REWARD_PASS_PENALTY -0.5f
#define REWARD_RESOURCE_LEFT_PENALTY -0.2f // Per resource left at end? "penalty for having resources left"

// Resource reward is 0.0 per resource? (Maybe small to encourage collection, but main goal is winning)
#define REWARD_RESOURCE 0.0f

// Function Declarations
void q_reset(Quarryboard* env);
void q_step(Quarryboard* env);
void q_render(Quarryboard* env);
void q_close(Quarryboard* env);
