/*
    Intention Repeater MAX CUDA v2.1 created by Thomas Sweet.
	CUDA, benchmark and flags functionality by Karteek Sheri.
    Created 11/12/2020 for C++.
	Requires: Visual Studio 2019 Community for C++: https://visualstudio.microsoft.com/downloads/
	Requires: CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
	Requires: Add location of cl.exe to Windows PATH.
	To compile: nvcc intention_repeater_max_cuda.cu -O 3 -o intention_repeater_max_cuda.exe
	For help: intention_repeater_max_cuda.exe --help
    Repeats your intention up to 1.0x10^18 times per second to make things happen.
    Intention Repeater MAX CUDA is powered by a Servitor (20 Years / 2000+ hours in the making) [HR 6819 Black Hole System].
    Servitor Info: https://enlightenedstates.com/2017/04/07/servitor-just-powerful-spiritual-tool/
    Website: https://www.intentionrepeater.com/
    Forum: https://forums.intentionrepeater.com/
    Licensed under GNU General Public License v3.0
    This means you can modify, redistribute and even sell your own modified software, as long as it's open source too and released under this same license.
    https://choosealicense.com/licenses/gpl-3.0/
*/

#include <string>

#include <cmath>

#include <iostream>

#include <ctime>

#include <ratio>

#include <chrono>

#include <iomanip>

#include <locale.h>

#include <vector>

#include <iterator>

#include <sstream>

using namespace std;
using namespace std::chrono;

#define USEGPU
#define ONE_MINUTE 60
#define ONE_HOUR 3600
std::string PROCESS_STATEMENT = " REGULATE/INTEGRATE/OM";

#ifdef USEGPU
//CUDA code added by Karteek Sheri
__device__ __managed__ unsigned long long int iterations = 0;

__global__ void intention_on_gpu(char * device_intention_value_array, const long long int num_chars) {

    for (long long int i = 0; i < num_chars; i++) {
        device_intention_value_array[i] = device_intention_value_array[i];
    }
	
    atomicAdd( & iterations, 1);
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

static
const char * short_scale[] = {
    "",
    "k",
    "M",
    "B",
    "T",
    "q",
	"Q",
	"s",
	"S"
};

static
const char * short_scale_hz[] = {
    "",
    "k",
    "M",
    "G",
    "T",
    "P",
	"E",
	"Z",
	"Y"
};

const char * scale(unsigned long long int n, int decimals = 1,
    const char * units[] = short_scale) {
    /*
     * Number of digits in n is given by
     * 10^x = n ==> x = log(n)/log(10) = log_10(n).
     *
     * So 1000 would be 1 + floor(log_10(10^3)) = 4 digits.
     */
    int digits = n == 0 ? 0 : 1 + floor(log10l(fabs(n)));

    // determine base 10 exponential
    int exp = digits <= 4 ? 0 : 3 * ((digits - 1) / 3);

    // normalized number
    double m = n / powl(10, exp);

    // no decimals? then don't print any
    if (m - static_cast < long > (n) == 0)
        decimals = 0;

    // don't print unit for exp<3
    static char s[64];
    static
    const char * fmt[] = {
        "%1.*lf%s",
        "%1.*lf"
    };
    sprintf(s, fmt[exp < 3], decimals, m, units[exp / 3]);
    return s;
}

const char * suffix(unsigned long long int n, int decimals = 1) {
    static char s[64];
    strcpy(s, scale(n, decimals, short_scale));
    return s;
}

const char * suffix_hz(unsigned long long int n, int decimals = 1) {
    static char s[64];
    strcpy(s, scale(n, decimals, short_scale_hz));

    return s;
}

#ifdef USEGPU
void setGPU(int desiredGPU) {

    int devicesCount;
    cudaGetDeviceCount( & devicesCount);
    if (desiredGPU < devicesCount) {
        cudaSetDevice(desiredGPU);
        std::cout << "GPU " << desiredGPU << " is selected." << std::endl;
    } else {
        std::cout << "GPU " << desiredGPU << ": No device found. Please check launch parameters." << std::endl;
        exit(0);
    }
}
#endif

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

void print_help() {
	cout << "Intention Repeater MAX CUDA v2.1 (c)2020 Thomas Sweet aka Anthro Teacher." << endl;
	cout << "CUDA and flags functionality by Karteek Sheri." << endl;
	cout << "Intention multiplying functionality by Thomas Sweet." << endl << endl;

	cout << "Optional Flags:" << endl;
	cout << "	a) --gpu or -g" << endl;
	cout << "	b) --dur or -d" << endl;
	cout << "	c) --imem or -m" << endl;
	cout << "	d) --intent or -i" << endl;
	cout << "	e) --help" << endl << endl;

	cout << "--gpu = GPU # to use. Default = 0." << endl;
	cout << "--dur = Duration in HH:MM:SS format. Example 00:01:00 to run for one minute. Default = \"Until Stopped.\"" << endl;
	cout << "--imem = Specify how many GB of GPU RAM to use. Default = 1.0. Higher amount produces a faster repeat rate, but takes longer to load into memory." << endl;
	cout << "--intent = Intention. Default = Prompt the user for intention." << endl;
	cout << "--help = Display this help." << endl << endl;

	cout << "Example automated usage: intention_repeater_max_cuda.exe --gpu 0 --dur 00:01:00 --imem 1.0 --intent \"I am calm.\"" << endl;
	cout << "Default usage: intention_repeater_max_cuda.exe" << endl << endl;

	cout << "gitHub Repository: https://github.com/tsweet77/repeater-max-cuda" << endl;
	cout << "Forum: https://forums.intentionrepeater.com" << endl;
	cout << "Website: https://www.intentionrepeater.com" << endl;	
}

std::string findsum(std::string a, std::string b){

    std::vector<int>finalsum; // Stores the final sum of two number
    
    int carry=0;  // Stores carry at each stage of calculation
    
    /* Step 1 starts here */
    
    int i = a.size()-1,j=b.size()-1; // Start adding from lowest significant bit
    while((i>=0) && (j>=0)){     // Loop until either of number exhausts first
        int x = (a[i]-'0') + (b[j]-'0') + carry; // Calculate the sum of digit in final sum by adding respective digits with previous carry.
        finalsum.push_back(x%10);  // Store the respective digit of the final sum in a vector.
        carry = x/10;                // update the carry. The carry for next step is the remaining number after forming the digit of final sum.
        i--;                        // Move one step towards the left in both the string(numbers)
        j--;
    }
    /*  Step 2 starts here */
    
    while(i>=0){          // If the number 1 was greater than number 2, then there must be some digits to be taken care off.
        int x = (a[i]-'0') + carry;  // Add the remaining digits to the carry one by one and store the unit digit.
        finalsum.push_back(x%10);
        carry = x/10;            // update the carry from each step.
        i--;
    }
    /* Step 3 starts here */
    
    while(j>=0){       // If the number 2 was greater than number 1, then there must be some digits to be taken care off.
        int x = (b[j]-'0') + carry; // Add the remaining digits to the carry one by one and store the unit digit.
        finalsum.push_back(x%10);
        carry = x/10;         // update the carry from each step.
        j--;
    }
    /* Step 4 starts here */
    
    while(carry){   //If after finishing addition of the two numbers, if there is still carry/leftover then we need to take it into the final sum.
        finalsum.push_back(carry%10); // Store digit one by one.
        carry=carry/10;       // Reduce carry
    }
    /* Step 5 starts here */
    std::stringstream final_iter;
  // Since vector pushes value at last, the most significant digits starts at the end of the vector. Thus print reverse.
        
        std::copy(finalsum.rbegin(), finalsum.rend(), std::ostream_iterator<int>(final_iter,""));
    
    return final_iter.str();
}

int main(int argc, char ** argv) {
    //std::setvbuf(stdout, NULL, _IONBF, 0); //Disable buffering to fix status display on some systems.

    std::string intention, ref_rate, intention_value, process_intention, duration, param_duration, param_intention, runtime_formatted;

    #ifndef USEGPU
    unsigned long long int iterations = 0;
    #endif
    int seconds = 0;
	unsigned long long int MULTIPLIER = 0;
    #ifdef USEGPU
    volatile int desiredGPU = 0;
    #endif

    float ram_size_value = 1 ;
    param_duration = "Until Stopped";
    ref_rate = "Average";
    param_intention = "";
    
    for(int i = 1; i < argc; i++){
		if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") ){
			print_help();
			exit(0);
        }else if(!strcmp(argv[i], "-g") || !strcmp(argv[i], "--gpu")){

            desiredGPU = atoi(argv[i+1]);
            
        }else if(!strcmp(argv[i], "-d") || !strcmp(argv[i], "--dur")){

			param_duration = argv[i+1];
        
        }else if(!strcmp(argv[i], "-r") || !strcmp(argv[i], "--rate")){
			
            ref_rate = argv[i+1];
        
        }else if(!strcmp(argv[i], "-m") || !strcmp(argv[i], "--imem")){
			
           ram_size_value = atof(argv[i+1]);
        
        }else if(!strcmp(argv[i], "-i") || !strcmp(argv[i], "--intent")){
			
            param_intention = argv[i+1];
		
        }else{
			if(i == argc-1){
				break;
			}
			std::cout<<"ERROR: Invalid Command Line Option Found: "<< argv[i]<< " Error "<<std::endl;
		} i++;
	}
    //cout<<param_duration<<" "<<ref_rate<<" "<<ram_size_value<<endl;             
    unsigned long long int INTENTION_MULTIPLIER = (ram_size_value*1024*1024*512);

    //std::locale comma_locale(std::locale(), new comma_numpunct());
    //std::cout.imbue(comma_locale);
	
    cout << "Intention Repeater MAX CUDA v2.1 created by Thomas Sweet aka Anthro Teacher." << endl;
    cout << "CUDA and flags functionality by Karteek Sheri." << endl;
	cout << "Intention multiplying functionality by Thomas Sweet." << endl;
    cout << "This software comes with no guarantees or warranty of any kind and is for entertainment purposes only." << endl;
    cout << "Press Ctrl-C to quit." << endl << endl;

    if ((param_intention) == "") {
        cout << "Intention: ";
        std::getline(std::cin, intention);

    } else {
        intention = param_intention;
    }
	
	cout << "Loading intention into memory." << std::flush;

	//Repeat string till it more than INTENTION_MULTIPLIER characters long.
	while (intention_value.length() < INTENTION_MULTIPLIER) {
		intention_value += intention;
		++MULTIPLIER;
	}
	
	--MULTIPLIER; //Account for having to reduce at the end.
	
	//Now, remove enough characters at the end to account for the process statement to limit to less than INTENTION_MULTIPLIER characters.
	unsigned long long int intention_value_length = intention_value.length();
	unsigned long long int intention_length = intention.length();
	int process_statement_length = PROCESS_STATEMENT.length();
	unsigned long long int intention_length_val = intention_value_length - intention_length - process_statement_length;
	
	intention_value = intention_value.substr(0,intention_length_val);
	intention_value += PROCESS_STATEMENT;

	cout << endl;
    
    #ifdef USEGPU
    setGPU(desiredGPU); // This is to set GPU
    #endif
    #ifdef USEGPU
    const unsigned long long int num_chars = intention_value.length();
    // declaring character array
    char * intention_value_array;
    intention_value_array = (char * ) malloc((num_chars + 1) * sizeof(char));
    char * device_intention_value_array;
    strcpy(intention_value_array, intention_value.c_str());

    cudaMalloc((void ** ) & device_intention_value_array, (num_chars + 1) * sizeof(char));
    cudaMemcpy(device_intention_value_array, intention_value_array, (num_chars + 1) * sizeof(char), cudaMemcpyHostToDevice);
    #endif

    duration = param_duration;
	
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    
    std::string iterations_string= "0",iterations_string_freq= "0";
        do {
            start = std::chrono::system_clock::now();
            while ((std::chrono::duration_cast < std::chrono::seconds > (end - start).count() != 1)) {
                    intention_on_gpu << < 65535, 1024 >>> (device_intention_value_array, num_chars);
                    cudaDeviceSynchronize();
                    end = std::chrono::system_clock::now();
            }
            
            ++seconds;
            runtime_formatted = FormatTimeRun(seconds);
            
            iterations_string_freq = to_string(iterations * MULTIPLIER);
            iterations_string= findsum(iterations_string, iterations_string_freq );
            
            int digits = iterations_string.length();
            int freq_digits = iterations_string_freq.length();

            std::cout<< "[" + runtime_formatted + "]" << " (" <<setprecision(3)<<fixed<<(stoull(iterations_string.substr(0,4)))/pow(10,3)<<"x10^"<<digits-1<<" / "<<(stoull(iterations_string_freq.substr(0,4))/pow(10,3))<<"x10^"<<freq_digits-1<<" Hz): "<<intention<<"     \r" << std::flush;
            iterations =0;
            
            if (runtime_formatted == duration) {
                std::cout << endl << std::flush;
                #ifdef USEGPU
                    cudaFree(device_intention_value_array);
                #endif
                exit(0);
            }
        } while (1);
}