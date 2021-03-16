/**
 * \file console.h
 * \brief a pack of functions to help with teh windows console
 * \version 1.0
 * \date 1/12/19
*/

#pragma once

// Basic includes
#include <iostream>
#include <string>
#include <Windows.h>

namespace console
{
	/**
	 * \brief sets the verbose status
	 * \param b a boolean to turn on/off the debug verbose
	*/
	void setVerbose(bool b);

	/**
	 * \brief Go to the (x, y) pos in the console
	*/
	void gotoxy(int x, int y);

	/**
	 * \brief Chane the color of the console
	 * \param foreground the color of the foreground
	 * \param background the color of the background
	*/
	void setColor(int foreground, int background);

	/**
	 * \brief tells there's an error that couldn't be recovered
	 * \param line the line of the error 
	*/
	void unrecoverableErrorMessage(std::string message);
	void unrecoverableErrorMessage(std::string message, int line);

	/**
	 * \brief tells there's an error that could be recovered
	 * \param line the line of the error
	*/
	void recoverableErrorMessage(std::string message);
	void recoverableErrorMessage(std::string message, int line);

	/**
	 * \brief tells a debug message if verbose if set to true
	 * \param line the line of the message
	*/
	void debugMessage(std::string message);
	void debugMessage(std::string message, int line);
}


