/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>

#include "engine_cpu.h"

engine * engine::create_cpu(const uint32_t p) { return new engine_cpu(p); }
