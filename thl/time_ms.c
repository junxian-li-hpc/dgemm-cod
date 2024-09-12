#include <sys/time.h>
double time_ms(void)
{
  struct timeval tm;
  double ms = 0.0;

  (void)gettimeofday(&tm, NULL);
  ms = tm.tv_sec * 1000.0;
  ms += tm.tv_usec / 1000.0;

  return ms;
}