/**
 * \file console.cpp
 * \brief a pack of functions to help with teh windows console
 * \version 1.0
 * \date 1/12/19
*/

// Personal includes
#include "console.h"

namespace console
{
	static bool VERBOSE = true;

	void setVerbose(bool b)
	{
		VERBOSE = b;
	}

	void gotoxy(int x, int y) {
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		COORD pos = { x, y };
		SetConsoleCursorPosition(hConsole, pos);
	}

	void setColor(int foreground, int background)
	{
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		if (foreground < 0 || foreground > 15)
		{
			recoverableErrorMessage("Foreground value not in [[0 ; 15]]");
			return;
		}
		if (background < 0 || background > 15)
		{
			recoverableErrorMessage("Background value not in [[0 ; 15]]");
			return;
		}
		SetConsoleTextAttribute(hConsole, foreground + background * 16);

	}

	void recoverableErrorMessage(std::string message)
	{
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO csInfo;
		GetConsoleScreenBufferInfo(hConsole, &csInfo);
		int oc = csInfo.wAttributes;

		setColor(12, 0);
		std::cout << message << std::endl;
		setColor(oc % 16, oc / 16);
	}

	void recoverableErrorMessage(std::string message, int line)
	{
		recoverableErrorMessage(message + std::to_string(line));
	}

	void unrecoverableErrorMessage(std::string message)
	{
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO csInfo;
		GetConsoleScreenBufferInfo(hConsole, &csInfo);
		int oc = csInfo.wAttributes;

		setColor(0, 12);
		std::cout << message << std::endl;
		setColor(oc % 16, oc / 16);
	}

	void unrecoverableErrorMessage(std::string message, int line)
	{
		unrecoverableErrorMessage(message + std::to_string(line));
	}


	void debugMessage(std::string message)
	{
		if (!VERBOSE)
			return;
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO csInfo;
		GetConsoleScreenBufferInfo(hConsole, &csInfo);
		int oc = csInfo.wAttributes;

		setColor(14, 0);
		std::cout << message << std::endl;
		setColor(oc % 16, oc / 16);
	}

	void debugMessage(std::string message, int line)
	{
		debugMessage(message + std::to_string(line));
	}
}