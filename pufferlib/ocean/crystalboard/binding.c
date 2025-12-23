#include "crystalboard.h"

#define Env Crystalboard
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Get seed from kwargs
    PyObject* seed_obj = PyDict_GetItemString(kwargs, "seed");
    if (seed_obj != NULL && PyLong_Check(seed_obj)) {
        env->seed = (int)PyLong_AsLong(seed_obj);
    } else {
        env->seed = 0;
    }
    
    // Get max_steps from kwargs (default 200)
    PyObject* max_steps_obj = PyDict_GetItemString(kwargs, "max_steps");
    if (max_steps_obj != NULL && PyLong_Check(max_steps_obj)) {
        env->max_steps = (int)PyLong_AsLong(max_steps_obj);
    } else {
        env->max_steps = 200;
    }

    // Get frameskip from kwargs (default 20)
    PyObject* frameskip_obj = PyDict_GetItemString(kwargs, "frameskip");
    if (frameskip_obj != NULL && PyLong_Check(frameskip_obj)) {
        env->frameskip = (int)PyLong_AsLong(frameskip_obj);
    } else {
        env->frameskip = 20;
    }

    // Get render_mode from kwargs (default 0)
    PyObject* render_mode_obj = PyDict_GetItemString(kwargs, "render_mode");
    if (render_mode_obj != NULL && PyLong_Check(render_mode_obj)) {
        env->render_mode = (int)PyLong_AsLong(render_mode_obj);
    } else {
        env->render_mode = 0;
    }

    // Get board_size from kwargs (default 10)
    PyObject* board_size_obj = PyDict_GetItemString(kwargs, "board_size");
    int board_size = 10;
    if (board_size_obj != NULL && PyLong_Check(board_size_obj)) {
        board_size = (int)PyLong_AsLong(board_size_obj);
    }
    
    env->width = board_size;
    env->height = board_size;
    env->board_tiles = env->width * env->height;
    
    // Allocate dynamic arrays
    env->board_owner = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    env->board_structure = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    env->board_feature = (unsigned char*)calloc(env->board_tiles, sizeof(unsigned char));
    
    if (!env->board_owner || !env->board_structure || !env->board_feature) {
        // Handle allocation failure?
        // In C extension, maybe set Python error?
        // For now just return non-zero?
        return -1;
    }

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
    return 0;
}

