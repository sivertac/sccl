#pragma once
#ifndef BINARY_UTIL_HEADER

#include <istream>

void print_stream_binary_xxd(std::istream &input);

void print_buffer_binary_xxd(const char *buffer, size_t size);

#endif