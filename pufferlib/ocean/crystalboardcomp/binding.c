#define PUFFER_BINDING
#include "crystalboardcomp.c"

#define Env CrystalboardComp
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* seed_obj = PyDict_GetItemString(kwargs, "seed");
    env->seed = (seed_obj && PyLong_Check(seed_obj)) ? (int)PyLong_AsLong(seed_obj) : 0;
    
    PyObject* max_steps_obj = PyDict_GetItemString(kwargs, "max_steps");
    env->max_steps = (max_steps_obj && PyLong_Check(max_steps_obj)) ? (int)PyLong_AsLong(max_steps_obj) : 200;

    PyObject* frameskip_obj = PyDict_GetItemString(kwargs, "frameskip");
    env->frameskip = (frameskip_obj && PyLong_Check(frameskip_obj)) ? (int)PyLong_AsLong(frameskip_obj) : 20;

    PyObject* render_mode_obj = PyDict_GetItemString(kwargs, "render_mode");
    env->render_mode = (render_mode_obj && PyLong_Check(render_mode_obj)) ? (int)PyLong_AsLong(render_mode_obj) : 0;

    PyObject* board_size_obj = PyDict_GetItemString(kwargs, "board_size");
    int board_size = 30; // Default larger for 4 players
    if (board_size_obj && PyLong_Check(board_size_obj)) {
        board_size = (int)PyLong_AsLong(board_size_obj);
    }
    
    env->width = board_size;
    env->height = board_size;
    env->board_tiles = env->width * env->height;
    
    env->board_owner = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    env->board_structure = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    env->board_feature = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    
    if (!env->board_owner || !env->board_structure || !env->board_feature) return -1;

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "crystals_connected", log->crystals_connected);
    assign_to_dict(dict, "valid_placements", log->valid_placements);
    assign_to_dict(dict, "invalid_placements", log->invalid_placements);
    
    // Per-player stats
    assign_to_dict(dict, "p1_crystal_collected", log->p1_crystal_collected);
    assign_to_dict(dict, "p2_crystal_collected", log->p2_crystal_collected);
    assign_to_dict(dict, "p3_crystal_collected", log->p3_crystal_collected);
    assign_to_dict(dict, "p4_crystal_collected", log->p4_crystal_collected);
    
    assign_to_dict(dict, "p1_crystals_connected", log->p1_crystals_connected);
    assign_to_dict(dict, "p2_crystals_connected", log->p2_crystals_connected);
    assign_to_dict(dict, "p3_crystals_connected", log->p3_crystals_connected);
    assign_to_dict(dict, "p4_crystals_connected", log->p4_crystals_connected);
    
    assign_to_dict(dict, "p1_win_rate", log->p1_win_rate);
    assign_to_dict(dict, "p2_win_rate", log->p2_win_rate);
    assign_to_dict(dict, "p3_win_rate", log->p3_win_rate);
    assign_to_dict(dict, "p4_win_rate", log->p4_win_rate);

    assign_to_dict(dict, "p1_return", log->p1_return);
    assign_to_dict(dict, "p2_return", log->p2_return);
    assign_to_dict(dict, "p4_return", log->p4_return);
    
    assign_to_dict(dict, "human_action_ready", log->human_action_ready);
    assign_to_dict(dict, "last_human_action", log->last_human_action);
    assign_to_dict(dict, "current_player", log->current_player);
    
    // Human Action Exposure (via environment pointer hack or expanded Log struct? 
    // The Log struct is copied from C to Python. We need to add fields to Log struct first in .h?
    // actually binding.c has access to `Env* env` but `my_log` only takes `Log* log`.
    // PufferLib's `vec_log` calls `my_log` with `env->log`. 
    // So we should add these fields to the `Log` struct in `crystalboardcomp.h` first!
    
    return 0;
}
