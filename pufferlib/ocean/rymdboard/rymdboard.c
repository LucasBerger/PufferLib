/* Rymdboard standalone executable for interactive play and testing.
 * 
 * Build: See scripts/build_ocean.sh rymdboard
 * Run: ./rymdboard
 * 
 * Play interactively by entering: card_idx x y rotation
 * Example: "4 3 3 0" places street card 4 at position (3,3) with rotation 0
 */

#include "rymdboard.h"

// Allocate buffers for standalone mode
static float observations[OBS_TOTAL_SIZE];
static int actions[1];
static float rewards[1];
static unsigned char terminals[1];

void play_interactive(Rymdboard* env) {
    char input[256];
    
    printf("\n========================================\n");
    printf("       RYMDBOARD - Interactive Mode      \n");
    printf("========================================\n\n");
    
    print_help();
    
    while (1) {
        print_board(env);
        print_market(env);
        print_crystals(env);
        
        printf("Step %d | Crystals: %d/3 | Last reward: %.2f\n", 
               env->tick, env->crystals_connected, rewards[0]);
        
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
            if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
                printf("Invalid position (must be 0-%d)\n", BOARD_SIZE - 1);
                continue;
            }
            if (rotation < 0 || rotation > 3) {
                printf("Invalid rotation (must be 0-3)\n");
                continue;
            }
            
            // Encode and execute action
            actions[0] = encode_action(card_idx, x, y, rotation);
            printf("\nPlacing card %d at (%d, %d) with rotation %d (action=%d)\n", 
                   card_idx, x, y, rotation, actions[0]);
            
            c_step(env);
            
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

void play_with_render(Rymdboard* env) {
    int selected_card = 0;
    int cursor_x = BOARD_SIZE / 2;
    int cursor_y = BOARD_SIZE / 2;
    int rotation = 0;
    
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_R)) {
            c_reset(env);
            rewards[0] = 0;
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
            cursor_x = (cursor_x - 1 + BOARD_SIZE) % BOARD_SIZE;
        }
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            cursor_x = (cursor_x + 1) % BOARD_SIZE;
        }
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            cursor_y = (cursor_y - 1 + BOARD_SIZE) % BOARD_SIZE;
        }
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
            cursor_y = (cursor_y + 1) % BOARD_SIZE;
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
            actions[0] = encode_action(selected_card, cursor_x, cursor_y, rotation);
            c_step(env);
        }
        
        // Render
        c_render(env);
        
        // Draw cursor and selection overlay
        BeginDrawing();
        
        // Draw cursor
        int cx = cursor_x * CELL_SIZE;
        int cy = cursor_y * CELL_SIZE;
        DrawRectangleLines(cx, cy, CELL_SIZE, CELL_SIZE, PUFF_WHITE);
        DrawRectangleLines(cx + 1, cy + 1, CELL_SIZE - 2, CELL_SIZE - 2, PUFF_WHITE);
        
        // Draw preview of selected card at cursor position
        Card* selected = (selected_card < NUM_BUILDING_SLOTS) 
            ? &env->building_market[selected_card]
            : &env->street_market[selected_card - NUM_BUILDING_SLOTS];
        Color preview_color = (selected_card < NUM_BUILDING_SLOTS) 
            ? (Color){187, 0, 187, 100}   // Purple with alpha
            : (Color){0, 187, 187, 100};  // Cyan with alpha
        draw_shape_preview(cx, cy, selected, rotation, preview_color, CELL_SIZE);
        
        // Draw selection info
        char info[128];
        snprintf(info, sizeof(info), "Card: %d | Pos: (%d,%d) | Rot: %d | WASD:move Q/E:rotate Space:place", 
                 selected_card, cursor_x, cursor_y, rotation);
        DrawText(info, 10, BOARD_SIZE * CELL_SIZE + 55, 12, PUFF_YELLOW);
        
        EndDrawing();
    }
}

int main(int argc, char* argv[]) {
    // Initialize environment
    Rymdboard env = {0};
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.seed = (argc > 1) ? atoi(argv[1]) : 42;
    env.max_steps = 200;
    env.frameskip = 20;
    
    // Reset to initial state
    c_reset(&env);
    
    // Check for render mode flag
    int render_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--render") == 0 || strcmp(argv[i], "-r") == 0) {
            render_mode = 1;
        }
    }
    
    if (render_mode) {
        env.render_mode = 1;
        printf("Starting Rymdboard with graphical rendering...\n");
        printf("Controls: 0-6 select card, WASD/arrows move, Q/E rotate, Space place, R reset\n");
        play_with_render(&env);
    } else {
        printf("Starting Rymdboard in console mode...\n");
        printf("Use '--render' or '-r' flag for graphical mode\n\n");
        play_interactive(&env);
    }
    
    c_close(&env);
    return 0;
}

