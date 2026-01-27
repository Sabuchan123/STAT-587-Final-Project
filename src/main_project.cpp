#include "../include/retrieve_tickers.h"
#include <iostream>
#include <filesystem>
#include <sstream>

int main() {
    // Ensuring python environment exists.
    std::filesystem::path cwd = std::filesystem::current_path();
    if (cwd.filename() != "STAT-587-Final-Project") while (cwd.filename() != "STAT-587-Final-Project") cwd = cwd.parent_path();
    #ifdef _WIN32
        std::filesystem::path env_dir = cwd / "PyScripts" / "env";
        std::filesystem::path env_pyt = env_dir / "Scripts" / "python.exe";
    #else
        std::filesystem::path env_dir = cwd / "PyScripts" / "env";
        std::filesystem::path env_pyt = env_dir / "bin" / "python3";
    #endif
    if (!std::filesystem::exists(env_pyt)) {
        std::cout << "Initializing virtual environment...\n";
        std::system("python -m venv ../PyScripts/env");
        if (std::filesystem::exists(cwd / "PyScripts" / "requirements.txt")) {
            std::cout << "Downloading dependencies...\n";
            #ifdef _WIN32
                std::system((cwd / "PyScripts\\env\\Scripts\\pip install -r ..\\PyScripts\\requirements.txt").string().c_str());
            #else
                std::system((cwd / "PyScripts/env/bin/pip install -r ../PyScripts/requirements.txt").string().c_str());
            #endif
        }
        std::cout << "Finished!\n";
    }
    else std::cout << "Virtual environment found.\n";

    std::vector<std::string> TICKERS = retrieve_tickers();

    std::stringstream ss;
    ss << "start /wait cmd /c " << env_pyt.string() << " -u \"" << cwd.string() << "/PyScripts/data_io.py\" ";
    for (int i = 0; i < TICKERS.size(); i++) ss << TICKERS[i] << " ";
    std::system(ss.str().c_str());

    ss.str("");
    ss.clear();
    ss << "start cmd /k " << env_pyt.string() << " -u \"" << cwd.string() << "/PyScripts/data_preprocessing.py\"";
    std::system(ss.str().c_str());

    return 1;
}