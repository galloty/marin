/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

#if defined(GPU)
#include "ocl.h"
#endif
#include "mersenne.h"

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		Mersenne::get_instance().quit();
	}

private:
#if defined(_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined(_WIN32)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() {}

	static application & get_instance()
	{
		static std::unique_ptr<application, deleter> instance(new application());
		return *instance;
	}

private:
	static std::string header(const std::vector<std::string> & args)
	{
		const char * const sys_ver =
#if defined(_WIN64)
			"win64";
#elif defined(_WIN32)
			"win32";
#elif defined(__linux__)
#if defined(__x86_64)
			"linux x64";
#elif defined(__aarch64__)
			"linux arm64";
#else
			"linux x86";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__)
			"macOS arm64";
#else
			"macOS x64";
#endif
#else
			"unknown";
#endif

		std::ostringstream comp_ver;
#if defined(__clang__)
		comp_ver << ", clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
		comp_ver << ", gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif

		const char * const ext =
#if defined(GPU)
			"";
#else
			"_cpu";
#endif

		std::ostringstream ss;
		ss << "marin" << ext << " version 25.08.4 (" << sys_ver << comp_ver.str() << ")" << std::endl;
		ss << "Copyright (c) 2025, Yves Gallot" << std::endl;
		ss << "marin is free source code, under the MIT license." << std::endl;
		ss << std::endl << "Command line: '";
		bool first = true;
		for (const std::string & arg : args)
		{
			if (first) first = false; else ss << " ";
			ss << arg;
		}
		ss << "'" << std::endl;
		return ss.str();
	}

private:
	static std::string usage()
	{
		const char * const ext =
#if defined(GPU)
			"";
#else
			"_cpu";
#endif
		std::ostringstream ss;
		ss << "Usage: marin" << ext << " [options]  options may be specified in any order" << std::endl;
		ss << "  -p <p>    exponent of the Mersenne number 2^p - 1 (3 <= p <= 1509949421)" << std::endl;
		ss << "  -LL       perform Lucas-Lehmer primality test" << std::endl;
#if defined(GPU)
		ss << "  -d <n>    set the device number (default 0)" << std::endl;
		ss << "  -h        validate hardware (quick Gerbicz-Li error checking for each size)" << std::endl;
#endif
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		std::cout << header(args) << std::endl;

		uint32_t p = 0;
		bool isLL = false;
		size_t device = 0;
#if defined(GPU)
		bool valid = false;
#endif
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-p")
			{
				const std::string str = ((arg == "-p") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				p = uint32_t(std::atoi(str.c_str()));
				bool isprime = (p % 2 != 0);
				for (uint32_t d = 3; p / d >= d; d += 2) if (p % d == 0) { isprime = false; break; }
				if (!isprime) throw std::runtime_error("p must be an odd prime.");
				if ((p < 3) || (p > 1509949421)) throw std::runtime_error("p is out of range.");
			}
			if (arg.substr(0, 3) == "-LL")
			{
				isLL = true;
			}
#if defined(GPU)
			if (arg.substr(0, 2) == "-d")
			{
				const std::string str = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(str.c_str()));
			}
			if (arg.substr(0, 2) == "-h")
			{
				valid = true;
			}
#endif
		}

		Mersenne & mersenne = Mersenne::get_instance();

#if defined(GPU)
 		if (valid) { mersenne.valid_gpu(device); return; }
#endif

		// Display info
		// mersenne.info(); return;

		// Internal test
		// mersenne.test(device); return;

		if (p == 0)
		{
			std::cout << usage() << std::endl;
#if defined(GPU)
			ocl::platform pfm;
			if (pfm.display_devices() == 0) throw std::runtime_error("No OpenCL device.");
#endif
			return;
		}

		if (isLL) mersenne.checkLL(p, device);
		else mersenne.check(p, device);
	}
};

int main(int argc, char * argv[])
{
	std::setvbuf(stderr, nullptr, _IONBF, 0);	// no buffer

	try
	{
		application & app = application::get_instance();
		app.run(argc, argv);
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << e.what() << std::endl << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
