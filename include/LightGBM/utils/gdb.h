/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Adapted from: https://www.codeproject.com/Articles/33249/Debugging-C-Code-from-Java-Application
 */
#ifndef LIGHTGBM_UTILS_GDB_H_
#define LIGHTGBM_UTILS_GDB_H_

#ifdef DEBUG
#define GDB() exec_gdb()
#define GDB_ON_TRIGGER() gdb_on_trigger()
#else
#define GDB() {}
#define GDB_BY_TRIGGER() {}
#endif

extern "C" void trigger_gdb();
extern "C" void signal_gdb_attached();

void gdb_on_trigger();

void exec_gdb();


#endif   // LightGBM_UTILS_GDB_H_
