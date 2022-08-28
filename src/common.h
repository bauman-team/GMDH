#pragma once
#ifdef GMDH_MODULE
    #include <pybind11/pybind11.h>
#endif

#ifdef __GNUC__
    #define likely(expr)    (__builtin_expect(!!(expr), 1))
    #define unlikely(expr)  (__builtin_expect(!!(expr), 0))
#else
    #define likely(expr)    expr
    #define unlikely(expr)  expr
#endif

#ifdef GMDH_LIB
    #define DISPLAYEDCOLORWARNING "\033[33m"
    #define DISPLAYEDCOLORINFO "\033[0m"
#endif
#define DISPLAYEDWARNINGMSG(expr, param) "\nWarning! The input " expr " is incorrect!\nThe default value is used (" param ")!\n"