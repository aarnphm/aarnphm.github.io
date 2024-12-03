/*
Note that in this version them motor control application is
responsible for setting up the buffer and as such should be
run first.

NOTE: ARGUMENTS FOR FUNCTIONS LISTED HAVE BEEN LEFT OUT FOR THE
MOST PART - YOU SHOULD FIGURE OUT HOW TO USE THE FUNCTIONS PROPERLY!
*/

// User Application
int main(int argc, char *argv[]) {
  FILE *fd;                       // Can also use int * fd;
  char *myFIFO = "FIFO DIRECTORY" // Consider using a custom directory in
                                  // /tmp/.. for example

  /*
  At this point you want to open the buffer using the
  fopen   function
  Once the buffer is open you can write to it with either
  fputs or write
  Once you have written to the buffer make sure to close it with
  fclose
  */
}

// Motor Control Application
int main(int argc, char *argv[]) {
  FILE *fd;                       // Can also use int * fd;
  char *myFIFO = "FIFO DIRECTORY" // Consider using a custom directory in
                                  // /tmp/.. for example
      char buf[BufferSize];

  /*
  First creat the FIFO using this series of function:
  unmask()
  mknod()
  fopen()

  Once the FIFO is set up you can read from it using
  fgets()
  After this close the buffer
  fclose()
  */
}

/*
This is a very bare-bones example of the FIFO buffer, but it should be
enough to get started. Consider using loops to improve the functionality
of the program.
*/
