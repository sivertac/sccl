
#include "BinaryUtil.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void printStreamBinaryXxd(std::istream &input) {
    const int bufferSize = 16;
    char buffer[bufferSize];

    int offset = 0;
    while (true) {
        input.read(buffer, bufferSize);
        int bytesRead = static_cast<int>(input.gcount());

        if (bytesRead == 0)
            break;

        std::cout << std::setw(8) << std::setfill('0') << std::hex << offset
                  << ": ";

        for (int i = 0; i < bufferSize; ++i) {
            if (i < bytesRead) {
                std::cout << std::setw(2) << std::setfill('0') << std::hex
                          << static_cast<int>(
                                 static_cast<unsigned char>(buffer[i]))
                          << ' ';
            } else {
                std::cout << "   ";
            }
        }

        std::cout << " ";

        for (int i = 0; i < bufferSize; ++i) {
            if (i < bytesRead) {
                char c = buffer[i];
                if (c >= 32 && c <= 126) {
                    std::cout << c;
                } else {
                    std::cout << '.';
                }
            } else {
                std::cout << " ";
            }
        }

        std::cout << std::endl;
        offset += bytesRead;
    }
}

void printBufferBinaryXxd(const char *buffer, size_t size) {
    std::istringstream input(std::string(buffer, size));
    printStreamBinaryXxd(input);
}
