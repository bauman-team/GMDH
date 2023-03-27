#pragma once
#ifdef GMDH_MODULE
    #include <pybind11/pybind11.h>
    #include <pybind11/iostream.h>
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

// defines for converting number to string
#define STREXPAND(x) #x
#define STR(x) STREXPAND(x)

// constants
#define MAXVERBOSENUMBER 1

// warnings messages
#define MINTHREADSWARNING(varName) "\nWarning: The value of '" varName "' can't be equal to 0 or a negative number other than -1. The invalid value has been replaced with the default value " varName "=1\n"
#define MAXTHREADSWARNING(varName) "\nWarning: The value of '" varName "' can't be greater than the number of supported concurrent threads. The invalid value has been replaced wtih " varName "=-1 to use the maximum possible number of threads\n"
#define MINVERBOSEWARNING(varName) "\nWarning: The value of '" varName "' can't be negative. The invalid value has been replaced with the default value " varName "=0\n"
#define MAXVERBOSEWARNING(varName) "\nWarning: The value of '" varName "' can't be greater than " STR(MAXVERBOSENUMBER) ". The invalid value has been replaced with " varName "=" STR(MAXVERBOSENUMBER) " to print the most detailed information\n"

// exceptions messages
#define OPENFILEEXCEPTION "The file can't be opened"
#define WRONGMODELFILEEXCEPTION(inputModel, realModel) "The expected model is " + realModel + " but the file contains " + inputModel + " model"
#define CORRUPTEDFILEEXCEPTION "The file is corrupted"

namespace GMDH {

/// @brief Class implementing file exceptions
class FileException : public std::exception {
    public:
    /** 
     * @brief Constructor (C strings).
     * 
     * @param message C-style string error message.
     *                 The string contents are copied upon construction.
     *                 Hence, responsibility for deleting the char* lies
     *                 with the caller. 
     */
    explicit FileException(const char* message)
        : msg_(message) {}

    /** 
     * @brief Constructor (C++ STL strings).
     * 
     * @param message The error message.
     */
    explicit FileException(const std::string& message)
        : msg_(message) {}

    /** 
     * @brief Destructor.
     * Virtual to allow for subclassing.
     */
    ~FileException() noexcept {}

    /** 
     * @brief Returns a pointer to the (constant) error description.
     * 
     * @return A pointer to a const char*. The underlying memory
     *          is in posession of the Exception object. Callers must
     *          not attempt to free the memory.
     */
    const char* what() const noexcept override {
       return msg_.c_str();
    }

protected:
    std::string msg_; //!< Error message
};
}