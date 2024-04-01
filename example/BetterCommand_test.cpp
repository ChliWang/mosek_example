//* how to use BetterCommand
//* a
//todo 1
//? ???
//! !!!

#include <iostream>

int main(int argc, char const *argv[]) 
{
    std::cout << "Number of command-line arguments: " << argc << std::endl;

    // 打印命令行参数
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    return 0;
}
// g++ main.cpp -o main
// ./main arg1 arg2 arg3

// Number of command-line arguments: 4
// Argument 0: ./main
// Argument 1: arg1
// Argument 2: arg2
// Argument 3: arg3
