import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


R0 = 15
gamma = 0.1
beta = R0 * gamma

S0 = 0.99
I0 = 0.01
R0_pop = 0.0
initial_conditions = [S0, I0, R0_pop]


t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)


solution = solve_ivp(
    sir_model, t_span, initial_conditions, args=(beta, gamma), t_eval=t_eval, method='RK45'
)


S = solution.y[0]
I = solution.y[1]
R = solution.y[2]
time = solution.t


plt.figure(figsize=(10, 6))
plt.plot(time, S, label='Susceptible (S)')
plt.plot(time, I, label='Infected (I)')
plt.plot(time, R, label='Recovered (R)')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.title('SIR Model of Measles Spread')
plt.legend()
plt.grid()
plt.show()