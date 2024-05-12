/*
    Intention Repeater MAX CUDA v5.23 (c)2020-2024 by Anthro Teacher aka Thomas Sweet.
    Updated 4/5/2024 by Anthro Teacher and Claude 3 Opus.
    To compile: nvcc -O3 Intention_Repeater_MAX_CUDA.cu -o Intention_Repeater_MAX_CUDA.exe -L/Users/tswee/miniconda3/Library/lib -lz
    Repeats your intention up to 100 PHz to make things happen.
    For help: intention_repeater_max_cuda.exe --help
    Intention Repeater MAX CUDA is powered by a Servitor (20 Years / 2000+ hours in the making) [HR 6819 Black Hole System].
    Servitor Info: https://enlightenedstates.com/2017/04/07/servitor-just-powerful-spiritual-tool/
    Website: https://www.intentionrepeater.com/
    Forum: https://forums.intentionrepeater.com/
    Licensed under GNU General Public License v3.0
    This means you can modify, redistribute and even sell your own modified software, as long as it's open source too and released under this same license.
    https://choosealicense.com/licenses/gpl-3.0/
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <locale.h>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <csignal>
#include <atomic>

std::atomic<bool> interrupted(false);

void signalHandler(int signum)
{
    //std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    interrupted.store(true);
}

// CUDA kernel to perform intention repeating and frequency updating
__global__ void intentionRepeaterKernel(const char *intentionMultiplied, unsigned long long int *freq, size_t intentionSize)
{
    unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intentionSize)
    {
        atomicAdd(freq, 1);
    }
}

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <sys/sysinfo.h>
#elif __APPLE__
#include <sys/sysctl.h>
#endif

#include "picosha2.h"
#include "zlib.h"

constexpr int ONE_MINUTE = 60;
constexpr int ONE_HOUR = 3600;

#ifdef _WIN32
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

constexpr int BLACK = 0;
constexpr int BLUE = 1;
constexpr int GREEN = 2;
constexpr int CYAN = 3;
constexpr int RED = 4;
constexpr int MAGENTA = 5;
constexpr int YELLOW = 6;
constexpr int WHITE = 7;
constexpr int DARKGRAY = 8;
constexpr int LIGHTBLUE = 9;
constexpr int LIGHTGREEN = 10;
constexpr int LIGHTCYAN = 11;
constexpr int LIGHTRED = 12;
constexpr int LIGHTMAGENTA = 13;
constexpr int LIGHTYELLOW = 14;
constexpr int LIGHTGRAY = 15;

const char *enum2str[] = {"BLACK", "BLUE", "GREEN", "CYAN", "RED", "MAGENTA", "YELLOW", "WHITE", "DARKGRAY", "LIGHTBLUE", "LIGHTGREEN", "LIGHTCYAN", "LIGHTRED", "LIGHTMAGENTA", "LIGHTYELLOW", "LIGHTGRAY"};
#elif __APPLE__
#define DEFAULT "\033[0m"
#define DARKGRAY "\033[1;30m"
#define BLACK "\033[0;30m"
#define LIGHTRED "\033[1;31m"
#define RED "\033[0;31m"
#define LIGHTGREEN "\033[1;32m"
#define GREEN "\033[0;32m"
#define LIGHTYELLOW "\033[1;33m"
#define YELLOW "\033[0;33m"
#define LIGHTBLUE "\033[1;34m"
#define BLUE "\033[0;34m"
#define LIGHTMAGENTA "\033[1;35m"
#define MAGENTA "\033[0;35m"
#define LIGHTCYAN "\033[1;36m"
#define CYAN "\033[0;36m"
#define WHITE "\033[1;37m"
#define LIGHTGRAY "\033[0;37m"
#else
#define DEFAULT "\033[0m"
#define DARKGRAY "\033[1;30m"
#define BLACK "\033[0;30m"
#define LIGHTRED "\033[1;31m"
#define RED "\033[0;31m"
#define LIGHTGREEN "\033[1;32m"
#define GREEN "\033[0;32m"
#define LIGHTYELLOW "\033[1;33m"
#define YELLOW "\033[0;33m"
#define LIGHTBLUE "\033[1;34m"
#define BLUE "\033[0;34m"
#define LIGHTMAGENTA "\033[1;35m"
#define MAGENTA "\033[0;35m"
#define LIGHTCYAN "\033[1;36m"
#define CYAN "\033[0;36m"
#define WHITE "\033[1;37m"
#define LIGHTGRAY "\033[0;37m"
#endif

const std::string HSUPLINK_FILE = "HSUPLINK.TXT";

class comma_numpunct : public std::numpunct<char>
{
protected:
    virtual char do_thousands_sep() const
    {
        return ',';
    }

    virtual std::string do_grouping() const
    {
        return "\03";
    }
};

unsigned long long int get_ninety_percent_free_memory()
{
    unsigned long long int free_memory = 0;

#ifdef _WIN32
    // Windows-specific memory information
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG freePhysMem = memInfo.ullAvailPhys;
    free_memory = static_cast<unsigned long long>(freePhysMem * 0.9); // 90% of free physical memory
#elif __linux__
    // Linux-specific memory information
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    unsigned long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;
    unsigned long long freePhysMem = memInfo.freeram;
    freePhysMem *= memInfo.mem_unit;
    free_memory = static_cast<unsigned long long>(freePhysMem * 0.9); // 90% of free memory
#elif __APPLE__
    // macOS-specific memory information
    int mib[2];
    int64_t physical_memory;
    size_t length;

    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);

    unsigned long long totalPhysMem = static_cast<unsigned long long>(physical_memory);
    unsigned long long freePhysMem = totalPhysMem * 0.9; // 90% of total memory
    free_memory = freePhysMem;
#else
    std::cerr << "Unsupported operating system" << std::endl;
    return static_cast<unsigned long long>(-1); // Return max value to indicate error
#endif

    return free_memory;
}

std::string getHSUPLINKContents()
{
    std::string hsuplink;
    std::string HSUPLINK_FILE = "HSUPLINK.TXT";
    std::ifstream hsuplinkFile(HSUPLINK_FILE);

    if (hsuplinkFile.is_open())
    {
        std::ostringstream hsuplinkStream;
        hsuplinkStream << hsuplinkFile.rdbuf();
        hsuplink = hsuplinkStream.str();
        hsuplinkFile.close();
    }
    else
    {
        return HSUPLINK_FILE;
    }
    // std::cout << "HSUPLINK.TXT contents: " << hsuplink << std::endl;

    std::ifstream intentionsFile("INTENTIONS.TXT");
    if (intentionsFile.is_open())
    {
        std::ostringstream intentionsStream;
        intentionsStream << intentionsFile.rdbuf();
        std::string intentions = intentionsStream.str();
        intentionsFile.close();

        size_t pos = 0;
        while ((pos = hsuplink.find("INTENTIONS.TXT", pos)) != std::string::npos)
        {
            hsuplink.replace(pos, 14, intentions);
            pos += intentions.length();
        }
    }
    // std::cout << "HSUPLINK.TXT contents: " << hsuplink << std::endl;
    return hsuplink;
}

std::string compressMessage(const std::string &message)
{
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));

    if (deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK)
    {
        return ""; // Compression initialization failed
    }

    zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(message.data()));
    zs.avail_in = message.size();

    std::string compressed;
    char outbuffer[32768]; // Output buffer
    int ret;
    do
    {
        zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
        zs.avail_out = sizeof(outbuffer);

        ret = deflate(&zs, Z_FINISH);

        if (compressed.size() < zs.total_out)
        {
            compressed.append(outbuffer, zs.total_out - compressed.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);

    if (ret != Z_STREAM_END)
    {
        return ""; // Compression failed
    }

    return compressed;
}

void readFileContents(const std::string &filename, std::string &intention_file_contents)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "File not found" << std::endl;
        std::exit(EXIT_FAILURE); // Terminate the program
    }

    std::ostringstream buffer;
    char ch;
    while (file.get(ch))
    {
        if (ch != '\0')
        {
            buffer.put(ch);
        }
    }

    intention_file_contents = buffer.str();
    file.close();
}

std::string display_suffix(std::string num, int power, std::string designator)
{
    if (power < 3)
    {
        return num;
    }

    std::string s;

    if (designator == "Iterations")
    {
        constexpr char iterations_suffix_array[] = {' ', 'k', 'M', 'B', 'T', 'q', 'Q', 's', 'S', 'O', 'N', 'D'};
        s = iterations_suffix_array[power / 3];
    }
    else // designator == "Frequency"
    {
        constexpr char frequency_suffix_array[] = {' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y', 'R'};
        s = frequency_suffix_array[power / 3];
    }

    std::string str2 = num.substr(0, power % 3 + 1) + "." + num.substr(power % 3 + 1, 3) + s;

    return str2;
}

std::string FormatTimeRun(int seconds_elapsed)
{
    int hour, min, sec;
    std::string hour_formatted, min_formatted, sec_formatted;

    hour = seconds_elapsed / ONE_HOUR;
    seconds_elapsed -= hour * ONE_HOUR;
    min = seconds_elapsed / ONE_MINUTE;
    seconds_elapsed -= min * ONE_MINUTE;
    sec = seconds_elapsed;

    hour_formatted = (hour < 10) ? "0" + std::to_string(hour) : std::to_string(hour);
    min_formatted = (min < 10) ? "0" + std::to_string(min) : std::to_string(min);
    sec_formatted = (sec < 10) ? "0" + std::to_string(sec) : std::to_string(sec);

    return hour_formatted + ":" + min_formatted + ":" + sec_formatted;
}

void print_color_help()
{
#ifndef _WIN32
    std::cout << WHITE << "Color values for flag: --color [COLOR]" << std::endl
              << std::endl;

    std::cout << DEFAULT << "DEFAULT" << std::endl;
    std::cout << DARKGRAY << "DARKGRAY" << std::endl;
    std::cout << BLACK << "BLACK" << std::endl;
    std::cout << LIGHTRED << "LIGHTRED" << std::endl;
    std::cout << RED << "RED" << std::endl;
    std::cout << LIGHTGREEN << "LIGHTGREEN" << std::endl;
    std::cout << GREEN << "GREEN" << std::endl;
    std::cout << LIGHTYELLOW << "LIGHTYELLOW" << std::endl;
    std::cout << YELLOW << "YELLOW" << std::endl;
    std::cout << LIGHTBLUE << "LIGHTBLUE" << std::endl;
    std::cout << BLUE << "BLUE" << std::endl;
    std::cout << LIGHTMAGENTA << "LIGHTMAGENTA" << std::endl;
    std::cout << MAGENTA << "MAGENTA" << std::endl;
    std::cout << LIGHTCYAN << "LIGHTCYAN" << std::endl;
    std::cout << CYAN << "CYAN" << std::endl;
    std::cout << WHITE << "WHITE" << std::endl;
    std::cout << LIGHTGRAY << "LIGHTGRAY" << std::endl;
#else
    SetConsoleTextAttribute(hConsole, WHITE);
    std::cout << "Color values for flag: --color [COLOR]" << std::endl
              << std::endl;

    for (int k = 1; k <= 15; k++)
    {
        SetConsoleTextAttribute(hConsole, k);
        std::cout << enum2str[k] << std::endl;
    }
#endif // _WIN32
}

void create_nesting_files()
{
    std::ofstream myfile;

    myfile.open("NEST-1.TXT");
    if (myfile.is_open())
    {
        for (int repnum = 1; repnum <= 10; repnum++)
        {
            myfile << "INTENTIONS.TXT\r\n";
        }
    }
    myfile.close();

    for (int filenum = 2; filenum <= 100; filenum++)
    {
        std::string filename = "NEST-" + std::to_string(filenum) + ".TXT";
        myfile.open(filename);
        for (int repnum = 1; repnum <= 10; repnum++)
        {
            myfile << "NEST-" + std::to_string(filenum - 1) + ".TXT\r\n";
        }
        myfile.close();
    }

    std::cout << "Intention Repeater Nesting Files Written." << std::endl;
    std::cout << "Be sure to have your intentions in the INTENTIONS.TXT file." << std::endl;
    std::cout << "To run with the nesting option, use --boostlevel 50, for example to use up to Nesting to 50 levels deep." << std::endl;
    std::cout << "--boostlevel valid values: 1 to 100." << std::endl;
    std::cout << "When using --boostlevel 50, for example, it will ignore the --intent, and use \"NEST-50.TXT\" for the intent instead." << std::endl
              << std::endl;
}

void print_help()
{
    const std::string helpText = R"(
Intention Repeater MAX CUDA v5.23 (c)2020-2024 by Anthro Teacher aka Thomas Sweet.
This utility repeats your intention millions of times per second, in computer memory, to aid in manifestation.
Performance benchmark, exponents and flags by Karteek Sheri.
Holo-Link framework by Mystic Minds. This implementation by Anthro Teacher.
Intention multiplying by Anthro Teacher.

Optional Flags:
 a) --dur or -d, example: --dur 00:01:00
 b) --imem or -m, example: --imem 5
 c) --intent or -i, example: --intent "I am love."
 d) --suffix or -s, example: --suffix HZ
 e) --color or -c, example: --color LIGHTBLUE
 f) --boostlevel or -b, example: --boostlevel 100
 g) --createnestingfiles or -p, example: --createnestingfiles
 h) --usehololink or -u
 i) --createhololinkfiles or -x
 j) --colorhelp or -n
 k) --restevery or -e
 l) --restfor or -r
 m) --compress or -x
 n) --hashing or -g
 o) --file or -f
 p) --file2 or -f2
 q) --help or -h

--dur = Duration in HH:MM:SS format. Default = Run until stopped manually.
--imem = Specify how many GB of System RAM to use. Higher amount repeats faster, but takes longer to load. Default = 1.0.
--intent = Intention. Default = Prompts the user for intention.
--suffix = Specify whether to show regular (Hz) designator or scientific notation (Exp). Default = HZ.
--usehololink = Utilize the Holo-Link framework by Mystic Minds. 
--createhololinkfiles will create the default Holo-Link files and exit. Run with this option before using the --usehololink option. This will overwrite the TXT files.
--color = Set the text color. Default = WHITE.
--colorhelp = List and show all available colors for text using the option --color.
--createnestingfiles = Create the NEST- files required for boosting.
--boostlevel = Set the level to boost the power (1-100). Use --createnestingfiles before using --boostlevel.
--restevery = Stop repeating every specified # of seconds.
--restfor = # of Seconds to rest for each rest period.
--compress = Use compression Default n.
--hashing = Use hashing. Default n.
--file = Specify file to use if applicable.
--file2 = Specify second file to use if applicable.
--help = Display this help.

Example usage:
intention_repeater_max.exe --dur "00:01:00" --imem 4.0 --intent "I am Love." --hashing y --compress y

Example usage with Holo-Link:
1) intention_repeater_max.exe --createhololinkfiles
2) intention_repeater_max.exe --usehololink --color LIGHTBLUE --suffix EXP --dur 00:01:00 --imem 4.0 --intent "I am calm."

Make sure to create your INTENTIONS.TXT file, in this folder, with your intentions, before running #2 above.  
The --intent option is ignored when using --usehololink, which instead uses the INTENTIONS.TXT file.

Example usage with Nesting Files:
1) intention_repeater_max.exe --createnestingfiles
2) intention_repeater_max.exe --color LIGHTBLUE --dur 00:01:00 --imem 4.0 --boostlevel 5

Make sure to create your INTENTIONS.TXT file, in this folder, with your intentions, and the Nesting Files before running #2 above.

gitHub Repository: https://github.com/tsweet77/repeater-max 
Forum: https://forums.intentionrepeater.com
Website: https://www.intentionrepeater.com
)";

#ifndef _WIN32
    std::cout << helpText << std::endl;
#else
    SetConsoleTextAttribute(hConsole, WHITE);
    std::cout << helpText << std::endl;
    SetConsoleTextAttribute(hConsole, WHITE);
#endif
}

std::string getBoostIntention(std::string param_boostlevel)
{
    // Convert param_boostlevel to an integer
    int boostLevel = std::stoi(param_boostlevel);
    std::ostringstream NESTING_FILE_CONTENTS;

    // Check if boostLevel is within the valid range (1 to 100)
    if (boostLevel < 1 || boostLevel > 100)
    {
        return "0";
    }

    // Loop from 1 to param_boostlevel
    for (int i = 1; i <= boostLevel; i++)
    {
        std::string fileName = "NEST-" + std::to_string(i) + ".TXT";
        std::ifstream file1(fileName);

        if (!file1.is_open())
        {
            return "0";
        }

        NESTING_FILE_CONTENTS << file1.rdbuf();
        file1.close();

        std::ifstream file2("INTENTIONS.TXT");
        if (!file2.is_open())
        {
            return "0";
        }

        NESTING_FILE_CONTENTS << file2.rdbuf();
        file2.close();
    }

    return NESTING_FILE_CONTENTS.str();
}

std::string FindSum(std::string a, std::string b)
{
    std::vector<int> finalsum;
    int carry = 0;

    int i = a.size() - 1;
    int j = b.size() - 1;
    while ((i >= 0) && (j >= 0))
    {
        int x = (a[i] - '0') + (b[j] - '0') + carry;
        finalsum.push_back(x % 10);
        carry = x / 10;
        i--;
        j--;
    }

    while (i >= 0)
    {
        int x = (a[i] - '0') + carry;
        finalsum.push_back(x % 10);
        carry = x / 10;
        i--;
    }

    while (j >= 0)
    {
        int x = (b[j] - '0') + carry;
        finalsum.push_back(x % 10);
        carry = x / 10;
        j--;
    }

    while (carry)
    {
        finalsum.push_back(carry % 10);
        carry = carry / 10;
    }

    std::stringstream final_iter;
    std::copy(finalsum.rbegin(), finalsum.rend(), std::ostream_iterator<int>(final_iter, ""));

    return final_iter.str();
}

void create_hololink_files()
{
    const std::string HOLOSTONE_FILE = "HOLOSTONE.TXT";
    const std::string THOUGHTFORM_A_FILE = "THOUGHTFORM_A.TXT";
    const std::string THOUGHTFORM_B_FILE = "THOUGHTFORM_B.TXT";
    const std::string AMPLIFIER_FILE = "AMPLIFIER.TXT";

    std::ostringstream HOLOLINK_CONTENTS;
    HOLOLINK_CONTENTS << "#Comments are designated with a # prefix, and such commands are to be ignored by the Holo-Link.\r\n"
                      << "#" << HSUPLINK_FILE << " CONFIG FILE v1.0\r\n"
                      << "#Holo-Link framework created by Mystic Minds (2022).\r\n"
                      << "#This implementation of the Holo-Link framework by Anthro Teacher.\r\n"
                      << "\r\n"
                      << "DECLARATION PRIMARY (Properties of thought forms and uplink):\r\n"
                      << "\r\n"
                      << "I declare the uplink multiply the energy received from the Holo-Stones by Infinity and densify all energy to the highest amount to achieve Instant Quantum Manifestation of the energetic programmings in "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << "I declare the Holo-Stones to funnel their energy into " << HOLOSTONE_FILE << ".\r\n"
                      << "\r\n"
                      << "I declare the Holo-Stones to amplify the power and receptivity of the energetic programmings in "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << "I declare the Holo-Stones to multiply the strength of the energetic programmings in "
                      << HSUPLINK_FILE << " and increase the potency at the most optimal rate.\r\n"
                      << "\r\n"
                      << "I declare that all energetic programmings in " << HSUPLINK_FILE
                      << " be imprinted, imbued and amplified with the new energy from the Holo-Stones.\r\n"
                      << "\r\n"
                      << HOLOSTONE_FILE << ", " << AMPLIFIER_FILE << ", " << THOUGHTFORM_A_FILE << " AND "
                      << THOUGHTFORM_B_FILE
                      << " are extremely pure and of highest vibration and are fully optimized for Instant Quantum Manifestation.\r\n"
                      << "\r\n"
                      << THOUGHTFORM_A_FILE
                      << " is creating an unbreakable and continuous connection and funnel energy to all energetic programmings in "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << THOUGHTFORM_A_FILE
                      << " uses energy from Infinite Source to continuously uphold a perfect link between the Holo-Stones and the "
                      << HSUPLINK_FILE
                      << " to bring in infinitely more energy into all energetic programmings in "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << THOUGHTFORM_B_FILE << " reinforces 100% of energy into all the energetic programmings in "
                      << HSUPLINK_FILE << " at the quantum level.\r\n"
                      << "\r\n"
                      << THOUGHTFORM_B_FILE
                      << " safely and efficiently removes all blockages in this system at the quantum level to allow for Instant Quantum Manifestation.\r\n"
                      << "\r\n"
                      << HOLOSTONE_FILE << " feeds " << AMPLIFIER_FILE
                      << " which amplifies the energy and feeds it back to " << HOLOSTONE_FILE
                      << " and repeats it to the perfect intensity.\r\n"
                      << "\r\n"
                      << "All energetic programmings listed in " << HSUPLINK_FILE
                      << " are now amplified to the highest power, speed and quantum-level precision using energy from the Holo-Stones which are sourced through "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << HOLOSTONE_FILE
                      << " works with Earth's Crystal Grid in the most optimal way possible for Instant Quantum Manifestation.\r\n"
                      << "\r\n"
                      << "Earth's Power Grid is extremely pure, cool, clean, efficient, optimized, and of highest vibration and is safely tapped in the most optimal way possible by HOLOSTONE.TXT for Instant Quantum Manifestation, and uses the least amount of electricity possible for everyone who desires this.\r\n"
                      << "UPLINK CORE (Reference any object, file, spell, etc. here):\r\n"
                      << "\r\n"
                      << HOLOSTONE_FILE
                      << " (Receives and distributes energy to all objects, files, spells, etc referenced below):\r\n"
                      << "\r\n"
                      << "[INSERT OBJECTS TO CHARGE]\r\n"
                      << "\r\n"
                      << "INTENTIONS.TXT\r\n"
                      << "\r\n"
                      << "DECLARATIONS SECONDARY (Add-ons that strengthen the properties of the uplink itself):\r\n"
                      << "\r\n"
                      << "I declare the Holo-Stones will uplink their energy into these energetic programmings in "
                      << HSUPLINK_FILE
                      << " to create instant, immediate and prominent results optimally, efficiently and effortlessly.\r\n"
                      << "\r\n"
                      << "I declare these energetic programmings in " << HSUPLINK_FILE
                      << " to grow stronger at the most optimal rate through the ever-growing power of the Holo-Stones.\r\n"
                      << "\r\n"
                      << "I call upon the Holo-Stones to channel the Atlantean Master Crystals, Infinite Source, Earth's Crystal Grid and Earth's Power Grid directly and utilize their energy as a funnel into HOLOSTONE.TXT which will then funnel into the energetic programmings in "
                      << HSUPLINK_FILE << ".\r\n"
                      << "\r\n"
                      << "The energetic programmings specified in " << HSUPLINK_FILE
                      << " are now being perfected and fully optimized.\r\n"
                      << "\r\n"
                      << "I declare that the more the energetic programmings in " << HSUPLINK_FILE
                      << " are used, the stronger they become.\r\n"
                      << "\r\n"
                      << "I am in my highest and most optimal reality/timeline.\r\n"
                      << "\r\n"
                      << "I am grounded, cleared, healed, balanced, strong-willed and I release what I do not need.\r\n"
                      << "\r\n"
                      << "Every day, in every way, it's getting better and better.\r\n"
                      << "\r\n"
                      << "The Atlantean Master Crystals AND Earth's Crystal Grid are open to Infinite Source.\r\n"
                      << "\r\n"
                      << "For my highest good and the highest good of all.\r\n"
                      << "\r\n"
                      << "Thank you. So be it. OM.\r\n"
                      << "ALL ABOVE STATEMENTS RESPECT THE FREE WILL OF ALL INVOLVED.\r\n";

    std::ofstream HOLOSTONE_FILE_FILE(HOLOSTONE_FILE);
    HOLOSTONE_FILE_FILE << "HOLOSTONE";
    std::ofstream THOUGHTFORM_A_FILE_FILE(THOUGHTFORM_A_FILE);
    THOUGHTFORM_A_FILE_FILE << "THOUGHTFORM A";
    std::ofstream THOUGHTFORM_B_FILE_FILE(THOUGHTFORM_B_FILE);
    THOUGHTFORM_B_FILE_FILE << "THOUGHTFORM B";
    std::ofstream AMPLIFIER_FILE_FILE(AMPLIFIER_FILE);
    AMPLIFIER_FILE_FILE << "AMPLIFIER";
    std::ofstream HSUPLINK_FILE_FILE(HSUPLINK_FILE);
    HSUPLINK_FILE_FILE << HOLOLINK_CONTENTS.str();

    std::cout << "Holo-Link files created." << std::endl;
    std::cout << "Remember to create your INTENTIONS.TXT file, in this folder, with all your intentions for the Holo-Link." << std::endl;
    std::cout << "You may do one to a line, or however you feel." << std::endl;
    std::cout << "You may now run with the --usehololink option." << std::endl;
    std::cout << "When using --usehololink, the option --intent, will be ignored, and INTENTIONS.TXT will be used instead." << std::endl;
    std::cout << "Good Luck!" << std::endl;
}

std::string MultiplyStrings(const std::string &num1, const std::string &num2)
{
    int len1 = num1.size();
    int len2 = num2.size();
    std::vector<int> result(len1 + len2, 0);

    for (int i = len1 - 1; i >= 0; --i)
    {
        for (int j = len2 - 1; j >= 0; --j)
        {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int sum = mul + result[i + j + 1];

            result[i + j + 1] = sum % 10;
            result[i + j] += sum / 10;
        }
    }

    std::string resultStr;
    for (int num : result)
    {
        if (!(resultStr.empty() && num == 0))
        {
            resultStr.push_back(num + '0');
        }
    }

    return resultStr.empty() ? "0" : resultStr;
}

int main(int argc, char **argv)
{
    std::signal(SIGINT, signalHandler);
    std::string intention, process_intention, intention_value, duration, param_duration;
    std::string param_intention, param_intention_2, param_boostlevel, param_color;
    std::string param_usehololink, runtime_formatted, ref_rate;
    std::string suffix_value = "HZ", HSUPLINK_FILE, param_restevery, param_restfor, param_file2 = "X";
    std::string param_compress, param_hashing, useHashing, useCompression, intention_hashed;
    std::string totalIterations = "0", totalFreq = "0", param_file = "X", intention_display = "", loading_message = "LOADING INTO MEMORY...";
    unsigned long long int multiplier = 0;
    unsigned long long int hashMultiplier = 0, freq = 0;
    int seconds = 0, restevery_int = 0, restfor_int = 0;
    int digits = 0, freqDigits = 0;
    float ram_size_value = 1;

    param_duration = "UNTIL STOPPED";
    param_intention = "X";
    param_color = "WHITE";
    param_usehololink = "NO";
    param_boostlevel = "0";
    param_restevery = "0";
    param_restfor = "0";
    param_hashing = "X";
    param_compress = "X";
    HSUPLINK_FILE = "HSUPLINK.TXT";

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            print_help();
            std::exit(EXIT_SUCCESS);
        }
        else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--colorhelp"))
        {
            print_color_help();
            std::exit(EXIT_SUCCESS);
        }
        else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--dur"))
        {
            param_duration = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--imem"))
        {
            ram_size_value = std::atof(argv[i + 1]);
        }
        else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--boostlevel"))
        {
            param_boostlevel = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--createnestingfiles"))
        {
            create_nesting_files();
            std::exit(EXIT_SUCCESS);
        }
        else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--intent"))
        {
            param_intention_2 = argv[i + 1];
            param_intention = (!strcmp(argv[i], "-u") || !strcmp(argv[i], "--usehololink")) ? HSUPLINK_FILE : param_intention_2;
        }
        else if (!strcmp(argv[i], "-u") || !strcmp(argv[i], "--usehololink"))
        {
            param_usehololink = "YES";
        }
        else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--color"))
        {
            param_color = argv[i + 1];
            std::transform(param_color.begin(), param_color.end(), param_color.begin(), ::toupper);
        }
        else if (!strcmp(argv[i], "-x") || !strcmp(argv[i], "--createhololinkfiles"))
        {
            create_hololink_files();
            std::exit(EXIT_SUCCESS);
        }
        else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--suffix"))
        {
            suffix_value = argv[i + 1];
            std::transform(suffix_value.begin(), suffix_value.end(), suffix_value.begin(), ::toupper);
        }
        else if (!strcmp(argv[i], "-e") || !strcmp(argv[i], "--restevery"))
        {
            param_restevery = argv[i + 1];
            restevery_int = std::stoi(param_restevery);
        }
        else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--restfor"))
        {
            param_restfor = argv[i + 1];
            restfor_int = std::stoi(param_restfor);
        }
        else if (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--hashing"))
        {
            param_hashing = argv[i + 1];
            std::transform(param_hashing.begin(), param_hashing.end(), param_hashing.begin(), ::toupper);
        }
        else if (!strcmp(argv[i], "-x") || !strcmp(argv[i], "--compress"))
        {
            param_compress = argv[i + 1];
            std::transform(param_compress.begin(), param_compress.end(), param_compress.begin(), ::toupper);
        }
        else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--file"))
        {
            param_file = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-f2") || !strcmp(argv[i], "--file2"))
        {
            param_file2 = argv[i + 1];
        }
    }

    if (param_boostlevel != "0")
    {
        intention = getBoostIntention(param_boostlevel);
        intention_display = "Using Nesting File Quantumly: NEST-" + param_boostlevel + ".TXT with INTENTIONS.TXT";
    }

    if (param_usehololink == "YES")
    {
        std::cout << "Loading HOLO-LINK Files..." << std::flush;
        intention = getHSUPLINKContents();
        intention_display = HSUPLINK_FILE;
    }

    unsigned long long int INTENTION_MULTIPLIER = static_cast<unsigned long long int>(ram_size_value * 1024 * 1024 * 512);
    unsigned long long int free_memory = get_ninety_percent_free_memory();

    if (free_memory != static_cast<unsigned long long>(-1))
    {
        if (free_memory < INTENTION_MULTIPLIER)
        {
            INTENTION_MULTIPLIER = free_memory;
        }
    }
    else
    {
        std::cout << "Error retrieving memory information." << std::endl;
        return 0;
    }

#ifndef _WIN32
    // Set the terminal color based on the --color flag.
    if (param_color == "DEFAULT")
        std::cout << DEFAULT << std::flush;
    else if (param_color == "BLACK")
        std::cout << BLACK << std::flush;
    else if (param_color == "RED")
        std::cout << RED << std::flush;
    else if (param_color == "GREEN")
        std::cout << GREEN << std::flush;
    else if (param_color == "YELLOW")
        std::cout << YELLOW << std::flush;
    else if (param_color == "BLUE")
        std::cout << BLUE << std::flush;
    else if (param_color == "MAGENTA")
        std::cout << MAGENTA << std::flush;
    else if (param_color == "CYAN")
        std::cout << CYAN << std::flush;
    else if (param_color == "LIGHTGRAY")
        std::cout << LIGHTGRAY << std::flush;
    else if (param_color == "DARK_GRAY")
        std::cout << DARKGRAY << std::flush;
    else if (param_color == "LIGHTRED")
        std::cout << LIGHTRED << std::flush;
    else if (param_color == "LIGHTGREEN")
        std::cout << LIGHTGREEN << std::flush;
    else if (param_color == "LIGHTYELLOW")
        std::cout << LIGHTYELLOW << std::flush;
    else if (param_color == "LIGHTBLUE")
        std::cout << LIGHTBLUE << std::flush;
    else if (param_color == "LIGHTMAGENTA")
        std::cout << LIGHTMAGENTA << std::flush;
    else if (param_color == "LIGHTCYAN")
        std::cout << LIGHTCYAN << std::flush;
    else if (param_color == "WHITE")
        std::cout << WHITE << std::flush;
#else
    if (param_color == "BLACK")
        SetConsoleTextAttribute(hConsole, BLACK);
    else if (param_color == "BLUE")
        SetConsoleTextAttribute(hConsole, BLUE);
    else if (param_color == "GREEN")
        SetConsoleTextAttribute(hConsole, GREEN);
    else if (param_color == "CYAN")
        SetConsoleTextAttribute(hConsole, CYAN);
    else if (param_color == "RED")
        SetConsoleTextAttribute(hConsole, RED);
    else if (param_color == "MAGENTA")
        SetConsoleTextAttribute(hConsole, MAGENTA);
    else if (param_color == "LIGHTYELLOW")
        SetConsoleTextAttribute(hConsole, LIGHTYELLOW);
    else if (param_color == "LIGHTGRAY")
        SetConsoleTextAttribute(hConsole, LIGHTGRAY);
    else if (param_color == "DARKGRAY")
        SetConsoleTextAttribute(hConsole, DARKGRAY);
    else if (param_color == "LIGHTBLUE")
        SetConsoleTextAttribute(hConsole, LIGHTBLUE);
    else if (param_color == "LIGHTGREEN")
        SetConsoleTextAttribute(hConsole, LIGHTGREEN);
    else if (param_color == "LIGHTCYAN")
        SetConsoleTextAttribute(hConsole, LIGHTCYAN);
    else if (param_color == "LIGHTRED")
        SetConsoleTextAttribute(hConsole, LIGHTRED);
    else if (param_color == "LIGHTMAGENTA")
        SetConsoleTextAttribute(hConsole, LIGHTMAGENTA);
    else if (param_color == "YELLOW")
        SetConsoleTextAttribute(hConsole, YELLOW);
    else if (param_color == "WHITE")
        SetConsoleTextAttribute(hConsole, WHITE);
#endif // Not Windows
    std::locale comma_locale(std::locale(), new comma_numpunct());
    std::cout.imbue(comma_locale);

    std::cout << "Intention Repeater MAX CUDA v5.23 (c)2020-2024" << std::endl;
    std::cout << "by Anthro Teacher aka Thomas Sweet." << std::endl
              << std::endl;

    std::string file_contents_original, file_contents, file_contents2_original, file_contents2, intention_original;

    if (param_boostlevel == "0" && param_usehololink == "NO")
    {
        if (param_intention == "X" && param_file == "X" && param_file2 == "X")
        {
            while (!interrupted)
            {
                std::cout << "Enter your Intention: ";
                if (!std::getline(std::cin, intention_original))
                {
                    // If getline fails (e.g., due to an interrupt), break out of the loop immediately
                    interrupted.store(true); // Ensure the flag is set if not already
                    return 0;
                }

                //std::cout << "Intention_original: " << intention_original << " intention_original.empty() " << intention_original.empty() << std::endl;

                if (!intention_original.empty())
                {
                    break; // Successfully got an intention, exit the loop
                }
                else if (!interrupted)
                {
                    // Only show the message if we're not interrupted
                    std::cout << "The intention cannot be empty. Please try again.\n";
                }
            }
            //intention_value = intention_original;
        }
        else
        {
            if (param_intention != "X") {
                intention_original = param_intention;
            }
            //intention_value = param_intention;
            intention_display = intention_original;
        }
    }

    if (param_file != "X" && param_boostlevel == "0" && param_usehololink == "NO")
    {
        // Open param_intent file and read the full file contents into intention
        readFileContents(param_file, file_contents_original);
        //intention_display += "Contents of: ";
    }
    if (param_file2 != "X" && param_boostlevel == "0" && param_usehololink == "NO")
    {
        // Open param_intent file and read the full file contents into intention
        readFileContents(param_file2, file_contents2_original);
        //intention_display = "Contents of: ";
    }

    size_t length1 = file_contents_original.size();
    size_t length2 = file_contents2_original.size();
    size_t length3 = intention_original.size();

    size_t max_length = (std::max)({length1, length2, length3});

    if (intention_original != "" && intention_original != "X" && param_boostlevel == "0" && param_usehololink == "NO")
    {
        // Normalize intention
        while (intention.length() < max_length)
        {
            intention += intention_original;
        }
        intention_display = intention_original;
    }

    if (param_file != "X" && param_boostlevel == "0" && param_usehololink == "NO")
    {
        // Normalize file_contents
        while (file_contents.length() < max_length)
        {
            file_contents += file_contents_original;
        }
        intention_display += "(" + param_file + ")";
    }
    if (param_file2 != "X" && param_boostlevel == "0" && param_usehololink == "NO")
    {
        // Normalize file_contents2
        while (file_contents2.length() < max_length)
        {
            file_contents2 += file_contents2_original;
        }
        intention_display += "(" + param_file2 + ")";
    }

    intention += file_contents + file_contents2;

    if (INTENTION_MULTIPLIER > 0)
    {
        // ...

        std::cout << loading_message << std::endl;
        std::string temp = intention;

        while (intention_value.length() < INTENTION_MULTIPLIER)
        {
            intention_value += intention;
            ++multiplier;
        }
        --multiplier; // Account for having to reduce at the end.

        long long int intention_value_length = intention_value.length();
        long long int intention_length = intention.length();
        long long int intention_length_val = intention_value_length - intention_length;

        intention_value = intention_value.substr(0, intention_length_val);
        digits = std::to_string(multiplier).length();
    }

    if (INTENTION_MULTIPLIER == 0)
    {
        intention_value = intention;
        multiplier = 1;
    }

    if (!interrupted && param_hashing == "X")
    {
        std::cout << "Use Hashing (y/N): ";
        if (!std::getline(std::cin, useHashing))
        {
            interrupted.store(true);
            if (interrupted)
            {
                // std::cerr << "Interrupted during hashing input. Exiting configuration.\n";
                return 0;
            }
        }
        transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);
    }
    else if (!interrupted)
    {
        useHashing = param_hashing;
        transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);
    }

    if (!interrupted && param_compress == "X")
    {
        std::cout << "Use Compression (y/N): ";
        if (!std::getline(std::cin, useCompression))
        {
            interrupted.store(true);
            if (interrupted)
            {
                // std::cerr << "Interrupted during compression input. Exiting configuration.\n";
                return 0;
            }
        }
        transform(useCompression.begin(), useCompression.end(), useCompression.begin(), ::tolower);
    }
    else if (!interrupted)
    {
        useCompression = param_compress;
        transform(useCompression.begin(), useCompression.end(), useCompression.begin(), ::tolower);
    }

    if (multiplier > 0)
    {
        std::cout << "Multiplier: " << display_suffix(std::to_string(multiplier), digits - 1, "Iterations") << std::endl;
    }

    if (useHashing == "y" || useHashing == "yes")
    {
        std::cout << "Hashing...          \r";
        intention_hashed = picosha2::hash256_hex_string(intention_value);
        if (INTENTION_MULTIPLIER > 0)
        {
            intention_value.clear();
            while (intention_value.length() < INTENTION_MULTIPLIER)
            {
                intention_value += intention_hashed;
                ++hashMultiplier;
            }
        }
        else
        {
            intention_value = intention_hashed;
            hashMultiplier = 1;
        }

        digits = std::to_string(hashMultiplier).length();
        std::cout << "Hash Multiplier: " << display_suffix(std::to_string(hashMultiplier), digits - 1, "Iterations") << std::endl;
    }
    else
    {
        hashMultiplier = 1;
    }

    long long int originalIntentionSize, compressedIntentionSize, compressionFactor;
    int compressionFactor_digits, compressedIntentionSize_digits, originalIntention_digits;

    if (useCompression == "y" || useCompression == "yes")
    {
        std::cout << "Compressing...          \r";
        originalIntentionSize = intention_value.length() * 2;
        intention_value = compressMessage(intention_value);
        compressedIntentionSize = intention_value.length() * 2;
        compressionFactor = (originalIntentionSize / compressedIntentionSize);

        compressionFactor_digits = std::to_string(compressionFactor).length();
        compressedIntentionSize_digits = std::to_string(compressedIntentionSize).length();
        originalIntention_digits = std::to_string(originalIntentionSize).length();

        std::cout << "Compression: " << display_suffix(std::to_string(compressionFactor), compressionFactor_digits - 1, "Iterations") << "X ["
                  << display_suffix(std::to_string(originalIntentionSize), originalIntention_digits - 1, "Frequency") << "B -> "
                  << display_suffix(std::to_string(compressedIntentionSize), compressedIntentionSize_digits - 1, "Frequency") << "B]     " << std::endl;
    }

    // Allocate memory on the GPU for intentionMultiplied and freq
    char *d_intentionMultiplied;
    unsigned long long int *d_freq;
    cudaMalloc(&d_intentionMultiplied, intention_value.size());
    cudaMalloc(&d_freq, sizeof(unsigned long long int));

    // Copy intentionMultiplied to the GPU
    cudaMemcpy(d_intentionMultiplied, intention_value.c_str(), intention_value.size(), cudaMemcpyHostToDevice);

    duration = param_duration;

    process_intention.reserve(intention_value.size() + 20);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    do
    {
        start = std::chrono::high_resolution_clock::now();
        end = std::chrono::high_resolution_clock::now();
        // Set freq to 0 on the GPU
        cudaMemset(d_freq, 0, sizeof(unsigned long long int));

        while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 1))
        {
            // Launch the CUDA kernel for intention repeating and frequency updating
            int blockSize = 256;
            int numBlocks = (intention_value.size() + blockSize - 1) / blockSize;
            intentionRepeaterKernel<<<numBlocks, blockSize>>>(d_intentionMultiplied, d_freq, intention_value.size());

            // Wait for the GPU to finish before accessing on host
            cudaDeviceSynchronize();

            // Copy the updated freq back to the CPU
            cudaMemcpy(&freq, d_freq, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            end = std::chrono::high_resolution_clock::now();
        }
        ++seconds;

        totalFreq = MultiplyStrings(std::to_string(freq), std::to_string(multiplier));
        totalFreq = MultiplyStrings(totalFreq, std::to_string(hashMultiplier));
        totalIterations = FindSum(totalIterations, totalFreq);
        digits = totalIterations.length();
        freqDigits = totalFreq.length();
        freq = 0;

        runtime_formatted = FormatTimeRun(seconds);

        if (suffix_value == "EXP")
        {
            std::cout << "[" + runtime_formatted + "]"
                      << " (" << std::setprecision(3) << std::fixed
                      << (std::stoull(totalIterations.substr(0, 4)) / std::pow(10, 3)) << "x10^" << digits - 1 << " / "
                      << (std::stoull(totalFreq.substr(0, 4)) / std::pow(10, 3)) << "x10^" << freqDigits - 1
                      << " Hz): " << intention_display << "     \r" << std::flush;
        }
        else // suffix_value = "HZ"
        {
            std::cout << "[" + runtime_formatted + "]"
                      << " (" << display_suffix(totalIterations, digits - 1, "Iterations") << " / "
                      << display_suffix(totalFreq, freqDigits - 1, "Frequency")
                      << "Hz): " << intention_display << "     \r" << std::flush;
        }

        if (runtime_formatted == duration || interrupted)
        {
            std::cout << std::endl
                      << std::flush;
            std::exit(EXIT_SUCCESS);
        }

        if ((restevery_int > 0) && (seconds % restevery_int == 0))
        {
            start = std::chrono::high_resolution_clock::now();
            end = std::chrono::high_resolution_clock::now();

            if (suffix_value == "EXP")
            {
                std::cout << "[" + runtime_formatted + "]"
                          << " (" << std::setprecision(3) << std::fixed
                          << (std::stoull(totalIterations.substr(0, 4)) / std::pow(10, 3)) << "x10^" << digits - 1 << " / "
                          << (std::stoull(totalFreq.substr(0, 4)) / std::pow(10, 3)) << "x10^" << freqDigits - 1
                          << " Hz): " << intention_display << "     \r" << std::flush;
            }
            else // suffix_value = "HZ"
            {
                std::cout << "[" + runtime_formatted + "]"
                          << " (" << display_suffix(totalIterations, digits - 1, "Iterations") << " / "
                          << display_suffix(totalFreq, freqDigits - 1, "Frequency")
                          << "Hz): " << intention_display << "     \r" << std::flush;
            }
            while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < restfor_int))
            {
                end = std::chrono::high_resolution_clock::now();
            }
        }
    } while (1);

    cudaFree(d_intentionMultiplied);
    cudaFree(d_freq);

#ifndef _WIN32
    std::cout << WHITE << std::flush;
#else
    SetConsoleTextAttribute(hConsole, WHITE);
#endif

    return 0;
}