
/*
    Intention Repeater MAX CUDA v1.0 created by Thomas Sweet.
	CUDA functions by Karteek Sheri.
    Created 11/2/2020 for C++.
	Requires: CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
	Requires: Visual Studio 2019 Community for C++: https://visualstudio.microsoft.com/downloads/
	Requires: Add location of cl.exe to Windows PATH.
	To compile: nvcc intention_repeater_max_cuda.cu -O 3 -o intention_repeater_max_cuda.exe
    Repeats your intention up to 500+ million times per second to make things happen.
    When compiled, this is more powerful than the Python version.
    Intention Repeater MAX is powered by a Servitor (20 Years / 2000+ hours in the making) [HR 6819 Black Hole System].
    Servitor Info: https://enlightenedstates.com/2017/04/07/servitor-just-powerful-spiritual-tool/
    Website: https://www.intentionrepeater.com/
    Forum: https://forums.intentionrepeater.com/
    Licensed under GNU General Public License v3.0
    This means you can modify, redistribute and even sell your own modified software, as long as it's open source too and released under this same license.
    https://choosealicense.com/licenses/gpl-3.0/
*/

#include <stdio.h>

#include <string>

#include <string.h>

#include <math.h>

#include <iostream>

#include <time.h>

#include <ctime>

#include <ratio>

#include <chrono>

#include <iomanip>

#include <locale.h>

#define USEGPU

using namespace std;
using namespace std::chrono;

std::string FormatTimeRun(int seconds_elapsed);

#define ONE_MINUTE 60
#define ONE_HOUR 3600
#define PROCESS_STATEMENT " REGULATE AND INTEGRATE. IT IS DONE. OM."

#ifdef USEGPU
    //CUDA code added by Karteek Sheri
    __device__ __managed__ unsigned long long int iterations = 0, frequency_count = 0;

    __global__ void intention_on_gpu(char *device_intention_value_array, const int num_chars){

        for(int i=0; i<num_chars;i++){
            device_intention_value_array[i] = device_intention_value_array[i];}
        atomicAdd(&iterations,1);
        atomicAdd(&frequency_count,1);

    }

#endif //end of CUDA code

class comma_numpunct: public std::numpunct < char > {
    protected: virtual char do_thousands_sep() const {
        return ',';
    }

    virtual std::string do_grouping() const {
        return "\03";
    }
};

static const char* short_scale[] = {
  "",
  "k",
  "M",
  "B",
  "T",
  "Q"
};

static const char* short_scale_hz[] = {
  "",
  "k",
  "M",
  "G",
  "T",
  "P"
};

const char* scale(double n, int decimals = 1, const char* units[] = short_scale)
{
  /*
   * Number of digits in n is given by
   * 10^x = n ==> x = log(n)/log(10) = log_10(n).
   *
   * So 1000 would be 1 + floor(log_10(10^3)) = 4 digits.
   */
  int digits = n==0? 0 : 1+floor(log10l(fabs(n)));

  // determine base 10 exponential
  int exp = digits <= 4? 0 : 3*((digits-1)/3);

  // normalized number
  double m = n / powl(10, exp);

  // no decimals? then don't print any
  if ( m - static_cast<long>(n) == 0 )
    decimals = 0;

  // don't print unit for exp<3
  static char s[32];
  static const char* fmt[] = {"%1.*lf%s", "%1.*lf"};
  sprintf(s, fmt[exp<3], decimals, m, units[exp/3]);
  return s;
}

const char* suffix(double n, int decimals = 1)
{
  static char s[32];
  strcpy(s, scale(n, decimals, short_scale));
  return s;
}

const char* suffix_hz(double n, int decimals = 1)
{
  static char s[32];
  strcpy(s, scale(n, decimals, short_scale_hz));

  return s;
}

int main(int argc, char ** argv) {
    std::string intention, intention_value, process_intention, duration, param_duration, param_intention, runtime_formatted;
    #ifndef USEGPU
        unsigned long long int iterations = 0, frequency_count = 0;
    #endif
    int seconds = 0;

    if (argc == 3) {
        param_duration = argv[1];
        param_intention = argv[2];

    } else {
        param_duration = "Until Stopped";
        param_intention = "";
    }

    std::locale comma_locale(std::locale(), new comma_numpunct());
    std::cout.imbue(comma_locale);

    cout << "Intention Repeater MAX CUDA v1.0 created by Thomas Sweet." << endl;
	cout << "CUDA functions by Karteek Sheri." << endl;
    cout << "This software comes with no guarantees or warranty of any kind and is for entertainment purposes only." << endl;
    cout << "Press Ctrl-C to quit." << endl << endl;

    if ((param_intention) == "") {
        cout << "Intention: ";
        std::getline(std::cin, intention);

    } else {
        intention = param_intention;
    }

    intention_value = intention + PROCESS_STATEMENT;
#ifdef USEGPU
    const int num_chars = intention_value.length();
	// declaring character array
    char* intention_value_array;
    intention_value_array = (char *)malloc((num_chars+1) * sizeof(char));
    //char device_intention_array[n+1];
    char *device_intention_value_array;
    strcpy(intention_value_array, intention_value.c_str());

	//for (int i = 0; i < num_chars; i++)
	//	cout << intention_value_array[i];
  
    cudaHostAlloc((void**)&device_intention_value_array,(num_chars+1)*sizeof(char),cudaHostAllocDefault);
    cudaMemcpy(device_intention_value_array,intention_value_array,(num_chars+1)*sizeof(char),cudaMemcpyHostToDevice);
#endif


    duration = param_duration;

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();

    do {
        start = std::chrono::system_clock::now();
		frequency_count = 0;
        while ((std::chrono::duration_cast < std::chrono::seconds > (end - start).count() != 1)) {
    #ifndef USEGPU
            process_intention = intention_value; //This is the Intention Repeater call that actually does the work with the Servitor.
            iterations += 1;
            frequency_count += 1;
    #else
            intention_on_gpu<<<65535,1024>>>(device_intention_value_array, num_chars);
            cudaDeviceSynchronize();
    #endif
            end = std::chrono::system_clock::now();
        }
        seconds += 1;
        runtime_formatted = FormatTimeRun(seconds);
        std::cout << "[" + runtime_formatted + "]" << " (" << suffix(iterations) << "/" << suffix_hz(frequency_count) << "Hz): " << intention << "     \r" << std::flush;
        if (runtime_formatted == duration) {
			std::cout << endl << std::flush;
            exit(0);
        }
    } while (1);
	std::cout << endl << "[" + runtime_formatted + "]" << " (" << suffix(iterations) << "|" << suffix_hz(frequency_count) << "Hz): " << intention << "     " << endl << std::flush;
    
	#ifdef USEGPU
		cudaFree(device_intention_value_array);
	#endif

    return 0;
}

std::string FormatTimeRun(int seconds_elapsed) {
    int hour, min, sec;

    std::string hour_formatted, min_formatted, sec_formatted;

    hour = seconds_elapsed / ONE_HOUR;
    seconds_elapsed -= hour * ONE_HOUR;
    min = seconds_elapsed / ONE_MINUTE;
    seconds_elapsed -= min * ONE_MINUTE;
    sec = seconds_elapsed;

    if (hour < 10) {
        hour_formatted = "0" + std::to_string(hour);
    } else {
        hour_formatted = std::to_string(hour);
    }

    if (min < 10) {
        min_formatted = "0" + std::to_string(min);
    } else {
        min_formatted = std::to_string(min);
    }

    if (sec < 10) {
        sec_formatted = "0" + std::to_string(sec);
    } else {
        sec_formatted = std::to_string(sec);
    }

    return hour_formatted + ":" + min_formatted + ":" + sec_formatted;
}