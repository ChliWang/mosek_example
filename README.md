# mosek_example
mosek for LP/CQO/POW/GP/SDO/MILO/MICO/DJC/Reopt
mosek link:https://docs.mosek.com/latest/cxxfusion/optimization-tutorials.html

# How To Use
```bash
mkdir build
cd build
cmake ..
make
./mosekTutorial_cqp
```

# by the way
If you are new to CMake, the way write CMakeLists.txt may help you.

# SOCP compare bet PLMALM and mosek
```
min a+2b+3c+4d+5e+6f+7g
||(7a + 1,6b + 3,5c+ 5,4d+ 7,3e+ 9,2f+ 11,g+ 13)|| ≤a+1.0
```
性能比较：常见的SOCP问题 (without using sparse matrix)  
- precise:1e-5  
- PLM-ALM+L-BFGS性能优于 Mosek，而Mosek更容易制定.  
- PLM-ALM+L-BFGS:0.00006800 second  
- Mosek Time: 0.00187487  
