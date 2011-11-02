#ifndef AUTOCALIB_CONFIG_H_
#define AUTOCALIB_CONFIG_H_

#define LOGGING_ENABLED 0

#if LOGGING_ENABLED
    #include <iostream>
    #define AUTOCALIB_LOG(x) x
#else
    #define AUTOCALIB_LOG(x)
#endif

#endif // AUTOCALIB_CONFIG_H_
