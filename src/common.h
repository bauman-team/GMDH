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
#define DISPLAYEDWARNINGMSG(expr, param) "\nWarning! The input " expr " is incorrect!\nThe default value is used (" param ")!\n"

#define GMDHPREDICTEXCEPTIONMSG "Input data number of cols is not match number of cols of fitted data!"
#define GMDHOPENFILEEXCEPTIONMSG "Input model file path is not exist!"
#define GMDHLOADMODELNAMEEXCEPTIONMSG(inputModel, realModel) "Input file for model: " inputModel ", but used model is " realModel "!"
#define GMDHLOADMODELPARAMSEXCEPTIONMSG "Input model file is corrupted!"

namespace GMDH {

class GmdhException : public std::exception {
    public:
    /** Constructor (C strings).
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction.
     *                 Hence, responsibility for deleting the char* lies
     *                 with the caller. 
     */
    explicit GmdhException(const char* message)
        : msg_(message) {}

    /** Constructor (C++ STL strings).
     *  @param message The error message.
     */
    explicit GmdhException(const std::string& message)
        : msg_(message) {}

    /** Destructor.
     * Virtual to allow for subclassing.
     */
    ~GmdhException() noexcept {}

    /** Returns a pointer to the (constant) error description.
     *  @return A pointer to a const char*. The underlying memory
     *          is in posession of the Exception object. Callers must
     *          not attempt to free the memory.
     */
    const char* what() const noexcept override {
       return msg_.c_str();
    }

protected:
    /** Error message.
     */
    std::string msg_;
};
}