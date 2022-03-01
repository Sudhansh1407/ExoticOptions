# Pricing Library for Exotic options
#
# Contents:
#    Part 1 (Closed-form solutions)
#       ->  Black-Scholes-Merton option pricing formula
#       ->  Simple Chooser Option
#       ->  Barrier Options
#       ->  Asian Geometric Options
#       ->  Digital Option
#       ->  Cash or Nothing Option
#       ->  Asset or Nothing Option
#
#    Part 2 (Simulation-based approximations)
#       ->  Lookback Options
#
#
# @author Sudhansh Dua
#
#
# Use the following code in your program after importing this file 
# (To load the new updates in the program):  
#                                            %load_ext autoreload
#                                            %autoreload 2









##############################################          Modules          ###############################################


import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from scipy.stats import t


##################################          Part 1: Closed-form solutions          #####################################


def Price_BSM(S, K, T, r, sig, typ = None, b = None):
    """
    Returns the price of the option based on the BSM formula for a European option
    
    Arguments:
    S- Spot Price
    K- Strike Price
    T- Time to maturity
    r- constant risk-free rate of return
    sig- constant volatility
    
    Cost of carry factor b must be included in formulae depending on the derivative type. 
    These are used in the generalised Black-Scholes formula.
    
    If r is the risk-free interest and q is the continuous dividend yield then the cost-of-carry b per derivative type is:
    a) Black-Scholes (1973) stock option model: b = r  
    b) b = r - q Merton (1973) stock option model with continuous dividend yield
    c) b = 0 Black (1976) futures option model
    d) b = r - rf Garman and Kohlhagen (1983) currency option model, where rf is the 'foreign' interest rate
    
    typ-  "C": Call 
          "P": Put;   typ = C ---> Default
    """
    
    # Default values
    if typ is None:
        typ = "C"
    if b is None:
        b = r
        
    d1 = (np.log(S/K)+(b+(sig*sig*0.5))*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    
    if (typ == "C"):
        # N(d1) = norm.cdf(d1)
        # N(d2) = norm.cdf(d2)
        # Call Price = S * e^(b-r)T * N(d1) - K * e^(-rT) * N(d2)
        return S*np.exp((b-r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)      
    else:
        # Put = K * e^(-rT) * N(-d2) - S * e^(b-r)T * N(-d1) 
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*exp((b-r)*T)*norm.cdf(-d1)
    
    
def SimpleChooser(S, K, t, T, r, sig, b=None):
    """
    A simple chooser option gives the holder the right to choose whether
    the option is to be a standard call or put after a time t , with strike K 
    and time to maturity T.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        t- time that has passed since t = 0
        T- Time to maturity at t = 0
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition
    if (t<T):
        
        # Default value
        if b is None: b = r

        y1 = (np.log(S/K) + b*T + (sig*sig*0.5*t)) / (sig * np.sqrt(t))
        y2 = y1 - sig*np.sqrt(t)

        d1 = (np.log(S/K) + b*T + sig*sig*0.5*T) / (sig * np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)

        w = S*(np.exp((b-r)*T)*norm.cdf(d1)) - K*(np.exp(-r*T)*norm.cdf(d2)) - S*(np.exp((b-r)*T)*norm.cdf(-y1)) + K*(np.exp(-r*T)*norm.cdf(-y2))
        return w
    
    else:
        raise TypeError("t cannot be greater than or equal to T")

        
        
def DownAndInCallBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Down-and-in call S > H 
        Payoff: max(S — K; 0) if S < H before T, else CR at expiration. 
        C(K>H) = C + E                         
        C(K<H) = A - B + D + E
            where ita = 1, phi = 1
        
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Down-and-in
    if (S>H):
        # Default value
        if b is None: b = r

        phi = 1
        ita = 1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return C + E
        else:
            return A - B + D + E
        
    else:
        raise TypeError("S cannot be less than H")
    

def DownAndInPutBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Down-and-in put S > H 
        Payoff: max(K — S; 0) if S < H before T else CR at expiration. 
        P(K>H) = B - C + D + E 
        P(K<H) = A + E
            where ita = 1, phi = -1
        
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Down-and-in
    if (S>H):
        # Default value
        if b is None: b = r

        phi = -1
        ita = 1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return B - C + D + E
        else:
            return A + E
    
    else:
        raise TypeError("S cannot be less than H")
        
        

def UpAndInCallBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Up-and-in call S < H 
        Payoff: max (S — K; 0) if S > H before T else CR at expiration. 
        C(K>H) = A + E 
        C(K<H) = B - C + D + E  
            where ita = -1, phi = 1
        
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Up-and-in
    if (S<H):
        
        # Default value
        if b is None: b = r

        
        phi = 1
        ita = -1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return A + E
        else:
            return B - C + D + E
        
    else:
        raise TypeError("S cannot be greater than or equal to H")
        
        
    
def UpAndInPutBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Up-and-in put S < H 
        Payoff: max(K — S; 0) if S > H before T else CR at expiration. 
        P(K>H) = A - B + D + E
        P(K<H) = C + E
            where ita = -1, phi = -1 
        
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Up-and-in
    if (S<H):
        
        # Default value
        if b is None: b = r

        phi = -1
        ita = -1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return A - B + D + E
        else:
            return C + E 
        
    else:
        raise TypeError("S cannot be greater than or equal to H")

        
def InBarrier(S, H, K, CR, T, r, sig, b = None, typ = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Down-and-in call S > H 
        Payoff: max(S — K; 0) if S < H before T, else CR at expiration. 
        C(K>H) = C + E                         
        C(K<H) = A - B + D + E
            where ita = 1, phi = 1
    
    Down-and-in put S > H 
        Payoff: max(K — S; 0) if S < H before T else CR at expiration. 
        P(K>H) = B - C + D + E 
        P(K<H) = A + E
            where ita = 1, phi = -1
        
    Up-and-in call S < H 
        Payoff: max (S — K; 0) if S > H before T else CR at expiration. 
        C(K>H) = A + E 
        C(K<H) = B - C + D + E  
            where ita = -1, phi = 1
        
    Up-and-in put S < H 
        Payoff: max(K — S; 0) if S > H before T else CR at expiration. 
        P(K>H) = A - B + D + E
        P(K<H) = C + E
            where ita = -1, phi = -1 
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default Values
    if b is None: b = r
    if typ is None: typ = "C"

    
    mu = (b - (sig*sig*0.5))/(sig*sig)
    psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
    x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
    x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)
    
    if (S>H):
        # Down-and-in
        ita = 1
        if (typ == "C"):
            # Down-and-in call
            phi = 1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H): 
                return C + E
            else: 
                return A - B + D + E
        else:
            # Down-and-in put
            phi = -1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            
            if (K>H):
                return B - C + D + E
            else:
                return A + E
         
    else:
        # Up-and-in
        ita = -1
        if (typ == "C"):
            # Up-and-in call
            phi = 1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H):
                return A + E
            else:
                return B - C + D + E
        else:
            # Up-and-in put
            phi = -1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H):
                return A - B + D + E
            else:
                return C + E
            
            
def DownAndOutCallBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate K, which is paid out if the option is knocked out before expiration.
    
    Down-and-out call S > H 
        Payoff: max(S — K; 0) if S > H before T else CR at hit. 
        C(X>H) = A - C + F
        C(x<H) = B - D + F
            where ita = 1, phi = 1
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Down-and-in
    if (S>H):
        
        # Default value
        if b is None: b = r

        phi = 1
        ita = 1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return A - C + F
        else:
            return B - D + F
    else:
        raise TypeError("S cannot be less than H")
    

    

def DownAndOutPutBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate K, which is paid out if the option is knocked out before expiration.
    
    Down-and-out put S > H 
        Payoff: max(K - S; 0) if S > H before T else CR at hit. 
        C(K>H) = A - B + C - D + F
        C(K<H) = F
            where ita = 1, phi = -1
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Down-and-in
    if (S>H):
        # Default value
        if b is None: b = r

        ita = 1
        phi = -1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return A - B + C - D + F
        else:
            return F
        
    else:
        raise TypeError("S cannot be less than H")
    
    
    
def UpAndOutCallBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate K, which is paid out if the option is knocked out before expiration.
    
    Up-and-out call S < H 
        Payoff: max(S — K; 0) if S < H before T else CR at hit. 
        C(K>H) = F 
        C(K<H) = A - B + C - D + F 
            where ita = -1, phi = 1
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Up-and-in
    if (S<H):
        # Default value
        if b is None: b = r

        ita = -1
        phi = 1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return F
        else:
            return A - B + C - D + F
    
    else:
        raise TypeError("S cannot be greater than or equal to H")
    
    
    
    
def UpAndOutPutBarrier(S, H, K, CR, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate K, which is paid out if the option is knocked out before expiration.
    
    Up-and-out put S < H 
        Payoff: max(K - S; 0) if S < H before T else CR at hit. 
        C(K>H) = B - D + F 
        C(K<H) = A - C + F 
            where ita = -1, phi = -1
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Condition for Up-and-in
    if (S<H):
        # Default value
        if b is None: b = r

        ita = -1
        phi = -1

        mu = (b - (sig*sig*0.5))/(sig*sig)
        psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
        x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
        x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K>H):
            return B - D + F
        else:
            return A - C + F
    else:
        raise TypeError("S cannot be greater than or equal to H")
    
    
    
    
def OutBarrier(S, H, K, CR, T, r, sig, b = None, typ = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate K, which is paid out if the option is knocked out before expiration.
    
    Down-and-out call S > H 
        Payoff: max(S — K; 0) if S > H before T else CR at hit. 
        C(X>H) = A - C + F
        C(x<H) = B - D + F
            where ita = 1, phi = 1
            
    Down-and-out put S > H 
        Payoff: max(K - S; 0) if S > H before T else CR at hit. 
        C(K>H) = A - B + C - D + F
        C(K<H) = F
            where ita = 1, phi = -1
            
    Up-and-out call S < H 
        Payoff: max(S — K; 0) if S < H before T else CR at hit. 
        C(K>H) = F 
        C(K<H) = A - B + C - D + F 
            where ita = -1, phi = 1
            
    Up-and-out put S < H 
        Payoff: max(K - S; 0) if S < H before T else CR at hit. 
        C(K>H) = B - D + F 
        C(K<H) = A - C + F 
            where ita = -1, phi = -1
    
    Arguments:
        S- Spot Price
        H- Barrier
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default Values
    if b is None: b = r
    if typ is None: typ = "C"
        
    # initialisation (temporary values)
    phi = 1
    ita = 1
    
    mu = (b - (sig*sig*0.5))/(sig*sig)
    psi = np.sqrt((mu*mu) + (2*r/(sig*sig)))
    x1 = (np.log(S/K)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)     
    x2 = (np.log(S/H)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    y1 = (np.log(H*H/(S*K))/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    y2 = (np.log(H/S)/(sig*np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
    z = (np.log(H/S)/(sig*np.sqrt(T))) + psi*sig*np.sqrt(T)

    
    if (S>H):
        # Down-and-out
        ita = 1
        if (typ == "C"):
            # Down-and-out call
            phi = 1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H): 
                return A - C + F
            else: 
                return B - D + F
        else:
            # Down-and-out put
            phi = -1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H):
                return A - B + C - D + F
            else:
                return F
         
    else:
        # Up-and-out
        ita = -1
        if (typ == "C"):
            # Up-and-out call
            phi = 1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H):
                return F
            else:
                return A - B + C - D + F
        else:
            # Up-and-out put
            phi = -1
            A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
            B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
            C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
            D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
            E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
            F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))
            if (K>H):
                return B - D + F
            else:
                return A - C + F
            
            
            
def AsianGeometricCall(S, K, T, r, sig, b = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed.
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> (Default)
    """
    # Default Values
    if b is None: b = r
    
    sig_a = sig / np.sqrt(3)
    b_a = 0.5 * (b - ((sig*sig)/6))
    
    d1 = (np.log(S/K)+(b_a+(sig_a*sig_a*0.5))*T)/(sig_a*np.sqrt(T))
    d2 = d1 - sig_a*np.sqrt(T)
    
    return S*np.exp((b_a-r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) 
    
    
def AsianGeometricPut(S, K, T, r, sig, b = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed.
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
    """
    # Default Values
    if b is None: b = r
        
    sig_a = sig / np.sqrt(3)
    b_a = 0.5 * (b - ((sig*sig)/6))
    
    d1 = (np.log(S/K)+(b_a+(sig_a*sig_a*0.5))*T)/(sig_a*np.sqrt(T))
    d2 = d1 - sig_a*np.sqrt(T)
    
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b_a-r)*T)*norm.cdf(-d1)


def AsianGeometric(S, K, T, r, sig, b = None, typ = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed.
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default Values
    if b is None: b = r
    if typ is None: typ = "C"
        
    sig_a = sig / np.sqrt(3)
    b_a = 0.5 * (b - ((sig*sig)/6))
    
    d1 = (np.log(S/K)+(b_a+(sig_a*sig_a*0.5))*T)/(sig_a*np.sqrt(T))
    d2 = d1 - sig_a*np.sqrt(T)
    
    if (typ == "C"):
        return S*np.exp((b_a-r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) 
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b_a-r)*T)*norm.cdf(-d1)



def DigitalOption(S, K, T, r, sig, typ = None, b = None):
    """
    The digital options pay $1 at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and 1 if S > K.
    The payoff from a put is 0 if S > K and 1 if S < K.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
        
    d = (np.log(S/K)+(b-(sig*sig*0.5))*T)/(sig*np.sqrt(T))
    
    if (typ == "C"):
        return np.exp(-r*T)*norm.cdf(d)
    else:
        return np.exp(-r*T)*norm.cdf(-d)


def CashOrNothing(S, K, CR, T, r, sig, typ = None, b = None):
    """
    The cash-or-nothing options pay an amount CR at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and CR, if S > K. 
    The payoff from a put is 0 if S > K and K if S < K.
    
    Arguments:
        S- Spot Price
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    d = (np.log(S/K)+(b-(sig*sig*0.5))*T)/(sig*np.sqrt(T))
    
    if (typ == "C"):
        return CR*np.exp(-r*T)*norm.cdf(d)
    else:
        return CR*np.exp(-r*T)*norm.cdf(-d)

        

def AssetOrNothing(S, K, T, r, sig, typ = None, b = None):
    """
    At expiration, the asset-or-nothing call option pays 0 if S < K and S if S > K. 
    Similarly, a put option pays 0 if S > K and S if S < K. 
    
    Arguments:
        S- Spot Price
        K- Strike Price
        CR- Cash Rebate
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
        
    d = (np.log(S/K)+((b+(sig*sig*0.5))*T))/(sig*np.sqrt(T))
    
    if (typ == "C"):
            return S*np.exp((b-r)*T)*norm.cdf(d)
    else:
            return S*np.exp((b-r)*T)*norm.cdf(-d)

        
        
#################################          Part 2: Simulation-based solutions          ####################################


def GBM_Simulation_Path(S0, T, r, sig, n, N, Z):
    """
    GBM:
    S1 = S0 * exp[(r - (sig*sig/2))*dt + sig * Z1]
    S2 = S1 * exp[(r - (sig*sig/2))*(2*dt) + sig * Z2]      or     S0 * exp[(r - (sig*sig/2))*(2*dt) + sig * (Z1 + Z2)]
    .......
    Sn = Sn-1 * exp[(r - (sig*sig/2))*(n*dt) + sig * Zn]    or     S0 * exp[(r - (sig*sig/2))*(n*dt) + sig * (Z1 + Z2 + ...... + Zn)]
    
    """

        
    zcum = np.cumsum(Z, axis=1)
    delta = T/N
    dt = np.array([delta*i for i in range(1, N+1)])
    f1 = r-0.5*sig**2
    f2 = sig*np.sqrt(delta)
    S = S0*np.exp(f1*dt+f2*zcum)
    return S

def Put_payoff(S, K):
    """
    The Payoff function for a Put Option
    
    Arguments:
    S- Spot Price
    K- Strike Price
    """
    return np.maximum(K-S, 0)

def Call_payoff(S, K):
    """
    The Payoff function for a Call Option
    
    Arguments:
    S- Spot Price
    K- Strike Price
    """
    return np.maximum(S-K, 0)








        


    
