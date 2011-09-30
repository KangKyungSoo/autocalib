#ifndef AUTOCALIB_CONFIG_H_
#define AUTOCALIB_CONFIG_H_

#define LOGGING_ENABLED 1

#if LOGGING_ENABLED
    #include <iostream>
    #define LOG(x) x
#else
    #define LOG(x)
#endif

#endif // AUTOCALIB_CONFIG_H_
