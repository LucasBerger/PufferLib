/* Crystalboard standalone executable for interactive play and testing.
 * 
 * Build: See scripts/build_ocean.sh crystalboard
 * Run: ./crystalboard [seed] [board_size] [--render]
 * 
 * Play interactively by entering: card_idx x y rotation
 * Example: "4 3 3 0" places street card 4 at position (3,3) with rotation 0
 */

#include "crystalboard.h"
#include <unistd.h>

// Allocate buffers for standalone mode
// Note: OBS_TOTAL_SIZE depends on board size... dynamic allocation needed for obs buffer too?
// Or just allocate a large enough buffer for testing.
// Max supported board size in standalone for now: 100x100?
#define MAX_TEST_BOARD 80
#define MAX_OBS_SIZE (MAX_TEST_BOARD*MAX_TEST_BOARD + 50000) 

static float observations[MAX_OBS_SIZE];
static int actions[1];
static float rewards[1];
static float total_reward = 0;
static unsigned char terminals[1];

void play_interactive(Crystalboard* env) {
    char input[256];
    
    printf("\n========================================\n");
    printf("       CRYSTALBOARD - Interactive Mode      \n");
    printf("       Board Size: %dx%d\n", env->width, env->height);
    printf("========================================\n\n");
    
    print_help();
    
    while (1) {
        print_board(env);
        print_market(env);
        print_crystals(env);
        
        printf("Step %d | Crystals: %d/3 | Last reward: %.2f | Total: %.2f\n", 
               env->tick, env->crystals_connected, rewards[0], total_reward);
        
        if (terminals[0]) {
            printf("\n*** EPISODE ENDED ***\n");
            if (env->crystals_connected >= 2) {
                printf("*** YOU WIN! Connected %d crystals! ***\n", env->crystals_connected);
            } else {
                printf("*** Game over (max steps or no valid moves) ***\n");
            }
            printf("Press 'r' to reset, 'q' to quit\n");
        }
        
        printf("\nCommand> ");
        fflush(stdout);
        
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // Parse input
        if (input[0] == 'q' || input[0] == 'Q') {
            printf("Goodbye!\n");
            break;
        }
        
        if (input[0] == 'r' || input[0] == 'R') {
            printf("Resetting game...\n");
            c_reset(env);
            rewards[0] = 0;
            total_reward = 0;
            terminals[0] = 0;
            continue;
        }
        
        if (input[0] == 'h' || input[0] == 'H') {
            print_help();
            continue;
        }
        
        // Parse action: card x y rotation
        int card_idx, x, y, rotation;
        if (sscanf(input, "%d %d %d %d", &card_idx, &x, &y, &rotation) == 4) {
            // Validate ranges
            if (card_idx < 0 || card_idx >= TOTAL_MARKET_SLOTS) {
                printf("Invalid card index (must be 0-%d)\n", TOTAL_MARKET_SLOTS - 1);
                continue;
            }
            if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
                printf("Invalid position (must be 0-%d)\n", env->width - 1);
                continue;
            }
            if (rotation < 0 || rotation > 3) {
                printf("Invalid rotation (must be 0-3)\n");
                continue;
            }
            
            // Encode and execute action
            actions[0] = encode_action(env, card_idx, x, y, rotation);
            printf("\nPlacing card %d at (%d, %d) with rotation %d (action=%d)\n", 
                   card_idx, x, y, rotation, actions[0]);
            
            c_step(env);
            total_reward += rewards[0];
            
            if (rewards[0] > 0) {
                printf(">>> Valid placement! Reward: %.2f\n", rewards[0]);
            } else if (rewards[0] < -1.0) {
                printf(">>> Invalid move! Reward: %.2f\n", rewards[0]);
            }
        } else {
            printf("Invalid input. Use: card x y rotation (e.g. '4 3 3 0')\n");
            printf("Or: 'r' to reset, 'q' to quit, 'h' for help\n");
        }
    }
}

void play_with_render(Crystalboard* env) {
    int selected_card = 0;
    int cursor_x = env->width / 2;
    int cursor_y = env->height / 2;
    int rotation = 0;
    
    // Calculate cell size same as c_render logic
    int max_dim = (env->width > env->height) ? env->width : env->height;
    int cell_size = 800 / max_dim;  // Reduced from 1000
    if (cell_size > 48) cell_size = 48;
    if (cell_size < 8) cell_size = 8;
    
    int market_width = 200;
    int win_w = env->width * cell_size + market_width;
    int win_h = env->height * cell_size + 80;
    
    // Initialize window
    if (!IsWindowReady()) {
        InitWindow(win_w, win_h, "Crystalboard - PufferLib Ocean");
        SetTargetFPS(60);
    }
    
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_R)) {
            c_reset(env);
            rewards[0] = 0;
            total_reward = 0;
            terminals[0] = 0;
        }
        
        // Card selection (number keys)
        for (int i = 0; i < TOTAL_MARKET_SLOTS; i++) {
            if (IsKeyPressed(KEY_ZERO + i) || IsKeyPressed(KEY_KP_0 + i)) {
                selected_card = i;
            }
        }
        
        // Cursor movement
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
            cursor_x = (cursor_x - 1 + env->width) % env->width;
        }
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            cursor_x = (cursor_x + 1) % env->width;
        }
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            cursor_y = (cursor_y - 1 + env->height) % env->height;
        }
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
            cursor_y = (cursor_y + 1) % env->height;
        }
        
        // Rotation
        if (IsKeyPressed(KEY_Q)) {
            rotation = (rotation - 1 + 4) % 4;
        }
        if (IsKeyPressed(KEY_E)) {
            rotation = (rotation + 1) % 4;
        }
        
        // Place (Enter or Space)
        if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {
            actions[0] = encode_action(env, selected_card, cursor_x, cursor_y, rotation);
            c_step(env);
            total_reward += rewards[0];
        }
        
        // Render in a single pass
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);
        
        // Draw the environment state
        c_draw_internal(env);
        
        // Draw cursor and selection overlay
        int cx = cursor_x * cell_size;
        int cy = cursor_y * cell_size;
        DrawRectangleLines(cx, cy, cell_size, cell_size, PUFF_WHITE);
        DrawRectangleLines(cx + 1, cy + 1, cell_size - 2, cell_size - 2, PUFF_WHITE);
        
        // Draw preview of selected card at cursor position
        Card* selected = (selected_card < NUM_BUILDING_SLOTS) 
            ? &env->building_market[selected_card]
            : &env->street_market[selected_card - NUM_BUILDING_SLOTS];
        Color preview_color = (selected_card < NUM_BUILDING_SLOTS) 
            ? (Color){187, 0, 187, 100}   // Purple with alpha
            : (Color){0, 187, 187, 100};  // Cyan with alpha
        draw_shape_preview(cx, cy, selected, rotation, preview_color, cell_size);
        
        // Draw selection info
        char info[128];
        snprintf(info, sizeof(info), "Card: %d | Pos: (%d,%d) | Rot: %d | Total Reward: %.2f", 
                 selected_card, cursor_x, cursor_y, rotation, total_reward);
        DrawText(info, 10, env->height * cell_size + 55, 12, PUFF_YELLOW);
        
        EndDrawing();
    }
}

int main(int argc, char* argv[]) {
    // Initialize environment
    Crystalboard env = {0};
    
    // Default settings
    int seed = 42;
    int board_size = 10;
    int render_mode = 0;
    
    // Parse args: ./crystalboard [seed] [board_size] [--render]
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--render") == 0 || strcmp(argv[i], "-r") == 0) {
            render_mode = 1;
        } else if (i == 1) {
            seed = atoi(argv[i]);
        } else if (i == 2 && seed != 0) { // If seed was parsed
            board_size = atoi(argv[i]);
        }
    }
    
    // Bounds check board size
    if (board_size < 5) board_size = 5;
    if (board_size > MAX_TEST_BOARD) board_size = MAX_TEST_BOARD;

    env.width = board_size;
    env.height = board_size;
    env.board_tiles = board_size * board_size;
    
    // Allocate dynamic arrays
    env.board_owner = (unsigned char*)calloc(env.board_tiles, sizeof(unsigned char));
    env.board_structure = (unsigned char*)calloc(env.board_tiles, sizeof(unsigned char));
    env.board_feature = (unsigned char*)calloc(env.board_tiles, sizeof(unsigned char));
    
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.seed = seed;
    env.max_steps = 200;
    env.frameskip = 20;

    printf("Initializing Crystalboard %dx%d with seed %d\n", board_size, board_size, seed);
    
    // Reset to initial state
    c_reset(&env);
    
#if defined(PLATFORM_WEB)
    render_mode = 1;
#endif

    if (render_mode) {
        env.render_mode = 1;
        env.frameskip = 1; // 60 FPS for standalone
        printf("Starting Crystalboard with graphical rendering...\n");
        printf("Controls: 0-6 select card, WASD/arrows move, Q/E rotate, Space place, R reset\n");
        play_with_render(&env);
    } else {
        printf("Starting Crystalboard in console mode...\n");
        printf("Use '--render' or '-r' flag for graphical mode\n\n");
        play_interactive(&env);
    }
    
    c_close(&env);
    return 0;
}

