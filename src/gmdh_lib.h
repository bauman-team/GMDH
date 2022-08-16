#pragma once

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
	// Windows Header Files
	#include <windows.h>
	#ifdef GMDH_LIB
		#ifdef GMDH_EXPORTS
			#define GMDH_API __declspec(dllexport)
		#else
			#define GMDH_API __declspec(dllimport)
		#endif
	#else
		#define GMDH_API
	#endif
#else
	#define GMDH_API
#endif

