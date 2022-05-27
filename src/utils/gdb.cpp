
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <cstdlib>

#include <iostream>
#include <sstream>



int _gdb_process_pid = 0;

using std::cout;
using std::endl;

volatile bool _trigger_gdb = false;

volatile bool _gdb_attached_signal = false;

extern "C" void trigger_gdb() {
  _trigger_gdb = true;
}

extern "C" void signal_gdb_attached() {
  _gdb_attached_signal = true;
}

void wait_for_gdb_attach() {
  while (!_gdb_attached_signal) {
    cout << "@parent> getpid()" << getpid() << endl;
    cout << "gdb_pid=" << _gdb_process_pid << "   <<<<<<<<<<\n\n" << endl;
    cout << "Sleeping to give GDB time to attach..." << endl;
    cout << "remember to call signal_gdb_attached()" << std::endl;
    sleep(10);
  }
  _gdb_attached_signal = false;
  cout << "Resuming..." << endl;
}

void exec_gdb2() {

  cout << "\n\n\n##### STARTING debugger #####\n\n" << endl;

  wait_for_gdb_attach();
}

void exec_gdb()
{
  cout << "\n\n\n##### STARTING debugger #####\n\n" << endl;


  int ref_pid = getpid();

  cout << "ref_pid=" << ref_pid << endl;

  // Create child process for running GDB debugger
  int pid = fork();

  if (pid < 0) /* error */
    {
      abort();
    }
  else if (pid) /* parent */
    {
      _gdb_process_pid = pid; // save debugger pid
      wait_for_gdb_attach();
      // Continue the application execution controlled by GDB
    }
  else /* child */
    {
      // GDB process. We run DDD GUI wrapper around GDB debugger

      std::stringstream args;
      // Pass parent process id to the debugger
      cout << "\n\n\nTrying to launch debugger with parent pid=" << getppid() << "..." << endl;
      args << "--pid=" << getppid();

      // Invoke debugger
      int debugger_rc = execl("/usr/bin/ddd", "ddd", "--debugger", "gdb", args.str().c_str(), (char *) 0);
      //execl("/usr/bin/gdb", "gdb", args.str().c_str(), (char *) 0);
      if (debugger_rc) {
        perror("The error message is:");
        std::cerr << "\nFailed to exec GDB (DDD)! Try attaching manually!\n" << endl;
        exit(1);
      }
      exit(0);
    }
}

void gdb_on_trigger() {
  if (_trigger_gdb) {
    exec_gdb();
    _trigger_gdb = false;
  }
}

