import numpy as np
from copy import deepcopy
from math import *

class Rocket:
    P_MAX = 5000          # максимальная тяга, Н
    OMEGA_SUM_MAX = 30    # максимальная суммарная масса топлива, кг
    ALPHA_MAX = 10        # максимальный угол атаки, град
    VEL_0 = 30            # начальная скорость
    I_1 = 2100            # удельный импульс топлива, м/с
    MASS_GOOD = 40        # масса полезной нагрузки, кг
    BETA = 1.3            # коэффициент массового совершенства двигательной установки
    D = 100e-3            # калибр, м
    
    
    def __init__(self, opts, alpha_foo):
        """
        opts - словарь с начальными данными 
            opts['theta'] = начальный угол тангажа в градусах (0 < theta < 90)
            opts['P1'] = тяга на стартовом участке в Ньютонах (0 < P1)
            opts['P2'] = тяга на маршевом участке в Ньютонах (0 < P2)
            opts['omega1'] = масса топлива на стартовом участке в кг (0 < omega1)
            opts['omega2'] = масса топлива на маршевом участке в кг (0 < omega2)
            
        alpha_foo - функция для определения потребного угла атаки в любой момент времени:
            принимает на вход следующие параметры:
            
                alpha_foo(t, v, x, y, theta, P, m, rho, M, Cya, Cx0, Sm)
                    t - текущее время, с
                    v - текущая скорость, м/с
                    x - текущая координата ЛА по оси х, м
                    y - текущая высота ЛА, м
                    theta - текущий угол тангажа в градусах
                    P - текущая тяга ЛА в Ньютонах
                    m - текущая масса ЛА в кг
                    rho - текущая плотность тмосферы, кг/м^3
                    M - текущее число Маха
                    Cya - текущее значение производной подъемной силы по углу атаки, 1/град
                    Cx0 - текущее значение коэффициента лобового сопротивления при нулевом уге атаки
                    Sm - площадь миделя, м^2                    
                    
                    
            функция alpha_foo должна возвращать занчение угла атаки в градусах -10 < alpha < 10
        """
        self.opts = deepcopy(opts)
        self.alpha_foo = alpha_foo
        
        self.init_atmo()
        self.init_ts_omegas(opts)
        self.init_CxCy()
    
    def reset(self):
        """
        Перевести ракету в начальное состояние.
        
        Возвращает t0, y0
        
        y0 = [v,   x, y, theta]
             [м/с, м, м, рад]
        """
        theta = self.opts['theta']
        assert 0 < theta <= 90
        self.statistics = []
        return 0, np.array([self.VEL_0, 0, 0, np.deg2rad(theta)]) 
    
    def get_dydt(self, t, y):
        """
        функция правых частей ОДУ
        t, c
        
        y =     [v,   x, y, theta]
                [м/с, м, м, рад]
            
        Возвращает
        dy/dt = [dv,    dx,  dy,  dtheta]
                [м/с^2, м/с, м/с, рад/с]
        """
        v, x, y, theta = y
        P = self.get_P(t)
        omega = self.get_omega(t)
        m = self.mass0 - omega
        rho = self.get_rho(y)
        a = self.get_a(y)
        M = v/a
        Cya = self.get_Cya(M)
        Cx0 = self.get_Cx0(M)
        if P < 0:
            P = 0
            Cx0 *= 1.05
        alpha = self.alpha_foo(t, v, x, y, np.rad2deg(theta), P, m, rho, M, Cya, Cx0, self.Sm)
        if alpha < -self.ALPHA_MAX: 
            alpha = -self.ALPHA_MAX
        if alpha > self.ALPHA_MAX:
            alpha = self.ALPHA_MAX
        Cx = Cx0 + (Cya * alpha) * np.tan(np.radians(alpha))
        g = 9.81
        self.statistics.append(
            (v, x, y, np.rad2deg(theta), P, m, rho, M, Cya, Cx0, alpha, Cx, t))
        return np.array([
            ( P * np.cos(np.radians(alpha)) - rho * v ** 2 / 2 * self.Sm * Cx - m * g * np.sin(theta) ) / m,
            v * np.cos(theta),
            v * np.sin(theta),
            ( alpha * ( Cya * rho * v ** 2 / 2 * self.Sm + P / 57.3) / ( m * g ) - np.cos(theta)) * g / v
        ], copy=False) 
        
    
    def init_ts_omegas(self, opts):
        """
        Инициировать массовые и тяговременные характеристики
        """
        
        self.P1 = opts['P1']
        self.P2 = opts['P2']
        self.omega1 = opts['omega1']
        self.omega2 = opts['omega2']
        assert 0 < self.P1 <= self.P_MAX 
        assert 0 < self.P2 <= self.P_MAX
        assert self.omega1 > 0
        assert self.omega2 > 0
        assert self.omega1 + self.omega2 <= self.OMEGA_SUM_MAX
        self.t1 = self.omega1 / self.P1 * self.I_1
        self.t2 = self.omega2 / self.P2 * self.I_1
        
        self.ts = np.array([0,self.t1,self.t1+self.t2])
        self.omegas = np.array([0,self.omega1, self.omega1+self.omega2])
        
        self.Ps = np.array([-1, self.P1, self.P2, -1])
        
        self.mass0 = self.MASS_GOOD + self.BETA * (self.omega1+self.omega2)
    
    def init_CxCy(self):
        """
        Инициировать "таблицы коэффициентов"
        """
        self.Ms = np.array([0.6,0.9,1.1,1.5,2])
        self.Cx0s = np.array([0.2,0.417,0.471, 0.924, 0.986])
        self.Cyas = np.array([0.306,0.341,0.246, 0.236, 0.218])
        self.Sm = np.pi * self.D **2 / 4
        
    def get_Cx0(self, M):
        """
        Получить интерполированное значение коэффициента лобового сопротивления при нулевом угле атаки 
        """
        return np.interp(M, self.Ms, self.Cx0s)
    
    def get_Cya(self, M):
        """
        Получить интерполированное значение производной коэффициента подъемной силы 
        """
        return np.interp(M, self.Ms, self.Cyas)
    
    def get_omega(self, t):
        """
        Получить массу сгоревшего топлива к моменту t, с
        """
        return np.interp(t, self.ts, self.omegas)
    
    def get_P(self, t):
        """
        Получить тягу к моменту t, с
        """
        i = np.searchsorted(self.ts,  t, side='right')
        return self.Ps[i]
      
    def init_atmo(self):
        """
        Инициировать параметры атмосферы
        https://tehtab.ru/Guide/GuidePhysics/Sound/SoundSpeedAirHeight/
        https://tehtab.ru/Guide/GuidePhysics/GuidePhysicsDensity/DensityAirHeight/
        """
        self.hs_cs = np.array([0,50,100,200,300,400,500,600,700,800,900,1000,5000,10000,20000,50000,80000])
        self.cs = np.array([340,340,339,339,339,338,338,337,337,337,336,336,320,299,295,329,382])
        self.hs_rhos = np.array([0,0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,12,15,20,50,100,120])
        self.rhos = np.array([1.225,1.219,1.213,1.202,1.19,1.167,1.112,1.007,0.909,0.736,0.526,0.414,0.312,0.195,0.089,1.027e-3,5.55e-7,2.44e-8])
    
        
    def get_a(self, h):
        """
        Получить скорость звука на высоте h, м
        """
        return np.interp(h, self.hs_cs, self.cs)

    def get_rho(self, h):
        """
        Плотность воздуха на высоте h, м
        """
        return np.interp(h/1000, self.hs_rhos, self.rhos)

    
def throw_foo(opts, alpha_foo, iter_max=130000, print_errors=True):
    """
    opts - словарь с начальными данными 
            opts['theta'] = начальный угол тангажа в градусах (0 < theta < 90)
            opts['P1'] = тяга на стартовом участке в Ньютонах (0 < P1)
            opts['P2'] = тяга на маршевом участке в Ньютонах (0 < P2)
            opts['omega1'] = масса топлива на стартовом участке в кг (0 < omega1)
            opts['omega2'] = масса топлива на маршевом участке в кг (0 < omega2)
            
        alpha_foo - функция для определения потребного угла атаки в любой момент времени:
            принимает на вход следующие параметры:
            
                alpha_foo(t, v, x, y, theta, P, m, rho, M, Cya, Cx0, Sm)
                    t - текущее время, с
                    v - текущая скорость, м/с
                    x - текущая координата ЛА по оси х, м
                    y - текущая высота ЛА, м
                    theta - текущий угол тангажа в градусах
                    P - текущая тяга ЛА в Ньютонах
                    m - текущая масса ЛА в кг
                    rho - текущая плотность тмосферы, кг/м^3
                    M - текущее число Маха
                    Cya - текущее значение производной подъемной силы по углу атаки, 1/град
                    Cx0 - текущее значение коэффициента лобового сопротивления при нулевом уге атаки
                    Sm - площадь миделя, м^2                    
                    
                    
            функция alpha_foo должна возвращать занчение угла атаки в градусах -10 < alpha < 10
    """
    rocket = Rocket(opts, alpha_foo)
    t, y = rocket.reset()
    dt = 0.05
    for i in range(iter_max):
        if y[2] < 0:
            break
        y1 = y + dt * rocket.get_dydt(t, y)
        t += dt
        if y1[1] < y[1]:
            if print_errors:
                print(f'Ошибка, мы летим назад)')
            break
        if y[0] < 30:
            if print_errors:
                print(f'Ошибка, мы слишком замедлились и потеряли управление')
            break
        y = y1
    else:
        if print_errors:
            print(f'Ошибка, мы совершили слишком много шагов интегрирования и так и не упали на землю')
    stats = np.array(rocket.statistics)
    return {
        'v': stats[:, 0], 
        'x': stats[:, 1], 
        'y': stats[:, 2], 
        'theta': stats[:, 3], 
        'P': stats[:, 4], 
        'm': stats[:, 5], 
        'rho': stats[:, 6], 
        'M': stats[:, 7], 
        'Cya': stats[:, 8], 
        'Cx0': stats[:, 9], 
        'alpha': stats[:, 10], 
        'Cx': stats[:, 11],
        't': stats[:, 12]
    }
