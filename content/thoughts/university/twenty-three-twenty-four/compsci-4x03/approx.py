import numpy as np, matplotlib.pyplot as plt

if __name__ == '__main__':
  dt = 0.1 # step size
  time = [0, 5] # [t0, tf]
  y0 = 0.5 # first value
  n_steps = int(np.ceil(time[1]/dt)) # tf/dt

  # time linear points
  t_points = np.linspace(time[0], time[1], n_steps+1)

  f = lambda y, t: y - t**2 + 1
  analytical = lambda t: (t+1)**2 - 0.5*np.exp(t) # solved by hand

  y_numerical = np.zeros(n_steps+1) # numerical value # array of [n_steps, 1]
  y_numerical[0] = y0

  for i in range(1, n_steps+1):
    y_numerical[i] = y_numerical[i-1] + f(y_numerical[i-1], t_points[i-1])*dt

  y_exact = analytical(t_points)

  # error
  max_err = np.max((err:=np.abs(y_exact - y_numerical)))
  print('max err:', max_err)

  plt.figure(figsize=(12,6))
  plt.plot(t_points, y_exact, 'b', label='analytical')
  plt.plot(t_points, y_numerical, 'r--', label='numerical')
  plt.title('comparison of numerical and analytical solutions')
  plt.xlabel('time x')
  plt.ylabel('y(x)')
  plt.legend()
  plt.grid(True)
  plt.show()
