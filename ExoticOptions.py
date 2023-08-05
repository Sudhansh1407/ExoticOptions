# Solutions for the pricing exotic derivatives
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
#       ->  Power Option
#
#    Part 2 (Simulation-based approximations)
#       ->  Digital option: Binomial Tree
#       ->  Digital option: Monte Carlo Simulation   
#       ->  
#
# @author Sudhansh Dua
#
#
# Use the following code in your program after importing this file 
# (To load the new updates in the program):  
#                                            %load_ext autoreload
#                                            %autoreload 2






####################################     Modules          ##################################


import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from scipy.stats import t


###############################         Part 1: Closed-form solutions     ##################


def price_BSM(S, K, T, r, sig, typ = None, b = None):
    """
    Returns the price of the option based on the BSM formula for a European option
    
    Arguments:
    --------------
    S : Spot Price
    K : Strike Price
    T : Time to maturity
    r : risk-free rate of return (constant)
    sig : volatility (constant)
    
    b : Cost of carry factor, must be included in formulae depending on the derivative type. These are used in the generalised Black-Scholes formula.
    
    If r is the risk-free interest and y is the continuous dividend yield then the cost-of-carry b for the derivative type will be given by:
    a) b = r      : Black-Scholes (1973) stock option model  
    b) b = r - q  : Merton (1973) stock option model with continuous dividend yield
    c) b = 0      : Black (1976) futures option model
    d) b = r - rf : Garman and Kohlhagen (1983) currency option model, where rf is the 'foreign' interest rate
    
    typ:  "C": Call 
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
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)      
    else:
        # Put = K * e^(-rT) * N(-d2) - S * e^(b-r)T * N(-d1) 
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * exp((b - r) * T) * norm.cdf(-d1)
    
    
def simple_chooser(S, K, t, T, r, sig, b=None):
    """
    A simple chooser option gives the holder the right to choose whether, at time t, the option will be a standard call or put with strike K and time to maturity T.
    
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 -----> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition
    if (t<T):
        
        # Default value
        if b is None: b = r

        y1 = (np.log(S/K) + b*T + (sig*sig*0.5*t)) / (sig * np.sqrt(t))
        y2 = y1 - sig * np.sqrt(t)

        d1 = (np.log(S/K) + b*T + sig*sig*0.5*T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)

        w = S*(np.exp((b-r)*T)*norm.cdf(d1)) - K*(np.exp(-r*T)*norm.cdf(d2)) - S*(np.exp((b-r)*T)*norm.cdf(-y1)) + K*(np.exp(-r*T)*norm.cdf(-y2))
        return w
    
    else:
        raise ValueError("t cannot be greater than or equal to T")

        
        
def down_and_in_barrier_call(S, H, K, CR =0, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Down-and-in call => S > H (starting spot price will be higher than the barrier)
    ----------------
        Payoff: max(S — K; 0) if S < H before T, else CR at expiration. 
        C(K > H) = C + E                         
        C(K < H) = A - B + D + E
            where ita = 1, phi = 1
        
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Down Barrier
    if (S > H):
        
        # Default values
        if b is None: b = r

        phi = 1
        ita = 1

        mu = (b - (sig * sig * 0.5))/(sig * sig)
        psi = np.sqrt((mu * mu) + (2 * r/(sig * sig)))
        x1 = (np.log(S / K) / (sig * np.sqrt(T))) + (1+ mu)* sig *np.sqrt(T)     
        x2 = (np.log(S / H) / (sig * np.sqrt(T))) + (1+ mu)* sig *np.sqrt(T)
        y1 = (np.log(H * H / (S * K)) / (sig * np.sqrt(T))) + (1+ mu) * sig * np.sqrt(T)
        y2 = (np.log(H / S) / (sig * np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H / S) / (sig * np.sqrt(T))) + psi * sig * np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K > H):
            return C + E
        else:
            return A - B + D + E
        
    else:
        raise ValueError("S cannot be less than H")
    

def down_and_in_barrier_put(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Down-and-in put => S > H (starting spot price will be higher than the barrier)
    ----------------
        Payoff: max(K — S; 0) if S < H before T else CR at expiration. 
        P(K > H) = B - C + D + E 
        P(K < H) = A + E
            where ita = 1, phi = -1
        
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Down Barrier
    if (S > H):
        # Default value
        if b is None: b = r

        phi = -1
        ita = 1
        
        mu = (b - (sig * sig * 0.5))/(sig * sig)
        psi = np.sqrt((mu * mu) + (2 * r/(sig * sig)))
        x1 = (np.log(S / K) / (sig * np.sqrt(T))) + (1+ mu)* sig *np.sqrt(T)     
        x2 = (np.log(S / H) / (sig * np.sqrt(T))) + (1+ mu)* sig *np.sqrt(T)
        y1 = (np.log(H * H / (S * K)) / (sig * np.sqrt(T))) + (1+ mu) * sig * np.sqrt(T)
        y2 = (np.log(H / S) / (sig * np.sqrt(T))) + (1+ mu)*sig*np.sqrt(T)
        z = (np.log(H / S) / (sig * np.sqrt(T))) + psi * sig * np.sqrt(T)

        A = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x1 - (sig*np.sqrt(T))))
        B = phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2) - phi*K*np.exp(-r*T)*norm.cdf(phi*(x2 - (sig*np.sqrt(T))))
        C = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y1) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y1 - (sig*np.sqrt(T))))
        D = phi*S*(pow(H/S,2*(mu+1)))*np.exp((b-r)*T)*norm.cdf(ita*y2) - phi*K*np.exp(-r*T)*(pow(H/S, 2*mu))*norm.cdf(ita*(y2 - (sig*np.sqrt(T))))
        E = CR*np.exp(-r*T)*(norm.cdf(ita*(x2-(sig*np.sqrt(T))))-(pow(H/S, 2*mu)*norm.cdf(ita*(y2-(sig*np.sqrt(T))))))
        F = CR*((pow(H/S, mu+psi)*norm.cdf(ita*z))+(pow(H/S, mu-psi))*norm.cdf(ita*(z-(2*psi*sig*np.sqrt(T)))))

        if (K > H):
            return B - C + D + E
        else:
            return A + E
    
    else:
        raise ValueError("S cannot be less than H")
        
        

def up_and_in_barrier_call(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset 
    price S hits the barrier H before expiration. It is possible to include 
    a prespecified cash rebate CR, which is paid out at option expiration if 
    the option has not been knocked in during its lifetime.
    
    Up-and-in call : S < H (starting spot price will be lower than the barrier)
    ---------------
        Payoff: max (S — K; 0) if S > H before T else CR at expiration. 
        C(K>H) = A + E 
        C(K<H) = B - C + D + E  
            where ita = -1, phi = 1
        
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Up Barrier
    if (S < H):
        
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
        raise ValueError("S cannot be greater than or equal to H")
        
        
    
def up_and_in_barrier_put(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The In options are paid for today but first come into existence if the asset price S hits the barrier H before expiration. It is possible to include a prespecified cash rebate CR, which is paid out at option expiration if the option has not been knocked in during its lifetime.
    
    Up-and-in put : S < H (starting spot price will be lower than the barrier)
    ---------------
        Payoff: max(K — S; 0) if S > H before T else CR at expiration. 
        P(K>H) = A - B + D + E
        P(K<H) = C + E
            where ita = -1, phi = -1 
        
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Up Barrier
    if (S < H):
        
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
        raise ValueError("S cannot be greater than or equal to H")

        
def in_barrier(S, H, K, CR = 0, T, r, sig, b = None, typ = None):
    """
    The In options are paid for today but first come into existence if the asset price S hits the barrier H before expiration. It is possible to include a prespecified cash rebate CR, which is paid out at option expiration if the option has not been knocked in during its lifetime.
    
    Down-and-in call : S > H 
    ---------------
        Payoff: max(S — K; 0) if S < H before T, else CR at expiration. 
        C(K>H) = C + E                         
        C(K<H) = A - B + D + E
            where ita = 1, phi = 1
    
    Down-and-in put : S > H
    ---------------
        Payoff: max(K — S; 0) if S < H before T else CR at expiration. 
        P(K>H) = B - C + D + E 
        P(K<H) = A + E
            where ita = 1, phi = -1
        
    Up-and-in call : S < H 
    ---------------
        Payoff: max (S — K; 0) if S > H before T else CR at expiration. 
        C(K>H) = A + E 
        C(K<H) = B - C + D + E  
            where ita = -1, phi = 1
        
    Up-and-in put : S < H 
    ---------------
        Payoff: max(K — S; 0) if S > H before T else CR at expiration. 
        P(K>H) = A - B + D + E
        P(K<H) = C + E
            where ita = -1, phi = -1 
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
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
            
            
def down_and_out_barrier_call(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate CR, which is paid out if the option is knocked out before expiration.
    
    Down-and-out call : S > H 
    ---------------
        Payoff: max(S — K; 0) if S > H before T else CR at hit 
        C(K > H) = A - C + F
        C(K < H) = B - D + F
            where ita = 1, phi = 1
            
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Down Barrier
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
        raise ValueError("S cannot be less than H")
    

    

def down_and_out_barrier_put(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate CR, which is paid out if the option is knocked out before expiration.
    
    Down-and-out put : S > H 
    ---------------
        Payoff: max(K - S; 0) if S > H before T else CR at hit 
        C(K>H) = A - B + C - D + F
        C(K<H) = F
            where ita = 1, phi = -1
    
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    
    # Condition for Down Barrier
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

        if (K > H):
            return A - B + C - D + F
        else:
            return F
        
    else:
        raise ValueError("S cannot be less than H")
    
    
    
def up_and_out_barrier_call(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate CR, which is paid out if the option is knocked out before expiration.
    
    Up-and-out call : S < H 
    ---------------
        Payoff: max(S — K; 0) if S < H before T else CR at hit
        C(K > H) = F 
        C(K < H) = A - B + C - D + F 
            where ita = -1, phi = 1
    
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Condition for Up Barrier
    if (S < H):
        
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

        if (K > H):
            return F
        else:
            return A - B + C - D + F
    
    else:
        raise ValueError("S cannot be greater than or equal to H")
    
    
    
def up_and_out_barrier_put(S, H, K, CR = 0, T, r, sig, b = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate CR, which is paid out if the option is knocked out before expiration.
    
    Up-and-out put : S < H 
    ---------------
        Payoff: max(K - S; 0) if S < H before T else CR at hit
        C(K>H) = B - D + F 
        C(K<H) = A - C + F 
            where ita = -1, phi = -1
    
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    
    """
    # Condition for Up Barrier
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
        raise ValueError("S cannot be greater than or equal to H")
    
    
    
    
def out_barrier(S, H, K, CR = 0, T, r, sig, b = None, typ = None):
    """
    The Out options are similar to standard options except that the option becomes worthless if the asset price S hits the barrier before expiration. 
    It is possible to include a prespecified cash rebate CR, which is paid out if the option is knocked out before expiration.
    
    Down-and-out call : S > H 
    ------------------
        Payoff: max(S — K; 0) if S > H before T else CR at hit 
        C(K>H) = A - C + F
        C(K<H) = B - D + F
            where ita = 1, phi = 1
            
    Down-and-out put : S > H
    -----------------
        Payoff: max(K - S; 0) if S > H before T else CR at hit 
        C(K>H) = A - B + C - D + F
        C(K<H) = F
            where ita = 1, phi = -1
            
    Up-and-out call : S < H
    -----------------
        Payoff: max(S — K; 0) if S < H before T else CR at hit 
        C(K>H) = F 
        C(K<H) = A - B + C - D + F 
            where ita = -1, phi = 1
            
    Up-and-out put : S < H 
    ----------------
        Payoff: max(K - S; 0) if S < H before T else CR at hit 
        C(K>H) = B - D + F 
        C(K<H) = A - C + F 
            where ita = -1, phi = -1
    
    Arguments:
    ---------------
        S : Spot Price
        H : Barrier
        K : Strike Price
        CR : Cash Rebate; CR = 0 ------> Default
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    
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
            
            
            
def asian_geometric_call(S, K, T, r, sig, b = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed.
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Default Values
    if b is None: b = r
    
    sig_a = sig / np.sqrt(3)
    b_a = 0.5 * (b - ((sig*sig)/6))
    
    d1 = (np.log(S/K)+(b_a+(sig_a*sig_a*0.5))*T)/(sig_a*np.sqrt(T))
    d2 = d1 - sig_a*np.sqrt(T)
    
    return S*np.exp((b_a-r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) 
    
    
def asian_geometric_put(S, K, T, r, sig, b = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed.
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
    """
    # Default Values
    if b is None: b = r
        
    sig_a = sig / np.sqrt(3)
    b_a = 0.5 * (b - ((sig*sig)/6))
    
    d1 = (np.log(S/K) + (b_a + (sig_a * sig_a * 0.5))*T)/ (sig_a * np.sqrt(T))
    d2 = d1 - sig_a*np.sqrt(T)
    
    return K * np.exp(-r*T)*norm.cdf(-d2) - S*np.exp((b_a-r)*T)*norm.cdf(-d1)


def asian_geometric(S, K, T, r, sig, b = None, typ = None):
    """
    If the underlying asset is assumed to be lognormally distributed, the geometric average ((s1 * s2 ..... * sn ) ^ 1/n) of the asset will itself be lognormally distributed. 
    
    The geometric average option can be priced as a standard option by changing the volatility and cost-of-carry term.
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
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



def digital_option(S, K, T, r, sig, typ = None, b = None):
    """
    The digital options pay $1 at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and 1 if S > K.
    The payoff from a put is 0 if S > K and 1 if S < K.
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
        
    d2 = (np.log(S/K) + (b - (sig*sig*0.5))*T)/(sig*np.sqrt(T))
    
    if (typ == "C"):
        # e^(-r*T) . N(d2)
        return np.exp(-r*T) * norm.cdf(d2)
    else:
        # e^(-r*T) . N(-d2)
        return np.exp(-r*T) * norm.cdf(-d2)


def cash_or_nothing(S, K, CR = 1, T, r, sig, typ = None, b = None):
    """
    The cash-or-nothing options pay an amount CR at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and CR, if S > K. 
    The payoff from a put is 0 if S > K and K if S < K.
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        CR- Cash Rebate; CR = 1 (default)
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default

    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    d2 = (np.log(S/K)+(b-(sig*sig*0.5))*T)/(sig*np.sqrt(T))
    
    if (typ == "C"):
        # C = CR . e^(-y*T) . N(d2)
        return CR * np.exp(-r * T) * norm.cdf(d2)
    else:
        # P = CR . e^(-y*T) . N(-d2)
        return CR * np.exp(-r * T) * norm.cdf(-d2)


def asset_or_nothing(S, K, T, r, sig, typ = None, b = None):
    """
    At expiration, the asset-or-nothing call option pays 0 if S < K and S if S > K. 
    Similarly, a put option pays 0 if S > K and S if S < K. 
    
    Arguments:
    ---------------
        S : Spot Price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
        
    d1 = (np.log(S/K) + ((b+(sig*sig*0.5))*T))/(sig*np.sqrt(T))
    
    if (typ == "C"):
        # C = S . e^(-y*T) . N(d1)
            return S * np.exp((b-r)*T) * norm.cdf(d1)
    else:
        # P = S . e^(-y*T) . N(-d1)
            return S * np.exp((b-r)*T) * norm.cdf(-d1)


        
def power_option_1(S, alpha, K, T, r, sig, typ = None, b = None):
    """
    A power option version 1, is like a normal option except the payoff is slightly different, for example, payoff of a power call at expiration = max(S_T^alpha - K, 0), where alpha can be any integer
    
    Arguments:
    ---------------
        S : Spot Price
        alpha : Exponent on Spot price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    S_eff = S**alpha
    sig_eff = np.abs(alpha)*sig
    # y_eff = r - alpha * ( b + ( alpha - 1)*0.5* sig**2 )
    b_eff = alpha * ( b + ( alpha - 1)*0.5* sig**2 )
    
    d1_eff = (np.log(S_eff/K) + ((b_eff+(0.5*sig_eff**2))*T))/(sig_eff*np.sqrt(T))
    d2_eff = d1_eff - sig_eff*np.sqrt(T)
    
    if (typ == "C"):
        # C = S^alpha . e^(-y_eff*T) . N(d1) - K . e^(-r*T) . N(d2)
            return S_eff * np.exp((b_eff - r) * T) * norm.cdf(d1_eff) - K * np.exp(-r * T) * norm.cdf(d2_eff)  
    else:
        # P  = K . e^(-r*T) . N(-d2) - [ S^alpha . e^(-y_eff*T) . N(-d1) ]
            return -S_eff * np.exp((b_eff - r) * T) * norm.cdf(-d1_eff) + K * np.exp(-r * T) * norm.cdf(-d2_eff) 


def power_option_2(S, alpha, K, T, r, sig, typ = None, b = None):
    """
    A power option version 2, is like a normal option except the payoff is slightly different, for example, payoff of a power call at expiration = S_T^alpha *[ max(S_T - K, 0) ], where alpha can be any integer
    
    Arguments:
    ---------------
        S : Spot Price
        alpha : Exponent on Spot price
        K : Strike Price
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    S_eff = S**alpha
    # sig_eff = sig
    
    r_eff = r - alpha * ( b + ( alpha - 1)*0.5* sig**2 )
    y_eff = y - alpha * ( b + ( alpha + 1)*0.5* sig**2 )
    b_eff = r_eff - y_eff
    
    d1_eff = (np.log(S/K) + ((b_eff+(0.5*sig_eff**2))*T))/(sig*np.sqrt(T))
    d2_eff = d1_eff - sig*np.sqrt(T)
    
    if (typ == "C"):
        # C = S^alpha [ S.e^(-y_eff*T) . N(d1) - K . e^(-r_eff*T) . N(d2)]
            return S_eff * (S * np.exp((b_eff - r_eff) * T) * norm.cdf(d1_eff) - K * np.exp(-r_eff * T) * norm.cdf(d2_eff) ) 
    else:
        # P  = S^alpha [ K . e^(-r_eff*T) . N(-d2) - [ S^alpha . e^(-y_eff*T) . N(-d1) ] ]
            return S_eff * (-S * np.exp((b_eff - r_eff) * T) * norm.cdf(-d1_eff) + K * np.exp(-r_eff * T) * norm.cdf(-d2_eff) )
    
########################          Part 2: Simulation-based solutions          ##########################


def stock_path_simulation(S0, T, r, sig, n, N, Z):
    """
    GBM:
    ----
    S1 = S0 * exp[(r - (sig*sig/2))*dt + sig * Z_1]
    S2 = S1 * exp[(r - (sig*sig/2))*(2*dt) + sig * Z_2]      or     S0 * exp[(r - (sig*sig/2))*(2*dt) + sig * (Z_1 + Z_2)]
    .......
    Sn = Sn-1 * exp[(r - (sig*sig/2))*(n*dt) + sig * Z_n]    or     S0 * exp[(r - (sig*sig/2))*(n*dt) + sig * (Z1 + Z2 + ...... + Z_n)]
    
    Arguments:
    ---------------
        S0 : Spot Price at t = 0
        T : Time to maturity
        r : risk-free rate of return (constant)
        sig : volatility (constant)
        n : no. of iterations
        N : no. of time intervals
        Z : Standard normal random variable N(0, 1)
    """

        
    zcum = np.cumsum(Z, axis=1)
    
    delta = T/N
    # array of time steps
    dt = np.array( [delta*i for i in range(1, N+1)] )
    
    drift = r - 0.5 * sig**2
    
    var_coeff = sig*np.sqrt(delta)
    
    S = S0 * np.exp(drift * dt + var_coeff *zcum)
    return S


def put_payoff(S, K):
    """
    The Payoff function for a Put Option
    
    Arguments:
    -----------
    S : Spot Price
    K : Strike Price
    """
    return np.maximum(K-S, 0)

def call_payoff(S, K):
    """
    The Payoff function for a Call Option
    
    Arguments:
    -----------
    S : Spot Price
    K : Strike Price
    """
    return np.maximum(S-K, 0)


def n_period_bin_model_uncond(S0, K, T, r, sig, n_steps = 10000, typ = "None", b = "None"):
    """
    The digital options pay $1 at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and 1 if S > K.
    The payoff from a put is 0 if S > K and 1 if S < K.
    
    Arguments:
        S0 : Spot Price at t = 0
        K : Strike Price
        CR : Cash Rebate
        T : Time to maturity
        r : constant risk-free rate of return
        sig : constant volatility
        N : number of periods/time steps; (if N = 2 and T = 1, one time step => 6 months)
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    delta_t = T/n_steps

    u = np.exp(sig * (delta_t ** 0.5))
    d = 1/u

    R = np.exp(r * delta_t)

    qu = (R - d)/(u - d)
    qd = 1 - qu

    
    stock_prices = []
    discounted_payoffs = []
    k = 0

    # defining n choose k function
    def comb(n, k):
        nCk = 1
        i = 1
        while i <= k:
            nCk *= (n - k + i)/i 
            i+=1
        return nCk
    
    while k < n_steps + 1:
        stock_price_k = S0 * (d ** k) * (u ** (n_steps - k)) 

        stock_prices.append(stock_price_k)

        if typ == "C":
            if stock_prices[k] > K:
                discounted_payoffs.append( comb(n_steps, k) * (qd ** k) * (qu ** (n_steps - k)) * np.exp(-r * T))
        else:
            if stock_prices[k] < K:
                discounted_payoffs.append( comb(n_steps, k) * (qd ** k) * (qu ** (n_steps - k)) * np.exp(-r * T))
        k+=1

    return sum(discounted_payoffs)




def digital_option_MC(S0, K, T, r, sig, M = 10000, typ = "None", b = "None"):
    """
    The digital options pay $1 at expiration if the option is in-the-money.
    The payoff from a call is 0 if S < K and 1 if S > K.
    The payoff from a put is 0 if S > K and 1 if S < K.
    
    Arguments:
        S0- Spot Price at t = 0
        K- Strike Price
        T- Time to maturity
        r- constant risk-free rate of return
        sig- constant volatility
        M- No. of simulations
        b- cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    W = np.random.randn(M, 1)
    prices = S0 * np.exp((r - (0.5 * sig **2)) * T + sig * np.sqrt(T) * W)

    p = prices - K
    payoffs = []
    for i in range(M):
        x = 1 if p[i] > 0 else 0
        payoffs.append(x) 
    

    if typ == "C":

        price = np.mean(payoffs) * np.exp(-r * T)
    else:
        # 1 - n/M = (M - n)/M where n is the number of times call digital was in the money
        price = (1 - np.mean(payoffs)) * np.exp(-r * T)

    return price


    
    
def option_MC(S0, K, T, r, sig, M = 10000, typ = "None", b = "None"):
    """
    The payoff from a call is 0 if S < K and S - K if S > K.
    The payoff from a put is 0 if S > K and S - K if S < K.
    
    Arguments:
    ------------
        S0 : Spot Price at t = 0
        K : Strike Price
        T : Time to maturity
        r : constant risk-free rate of return
        sig : constant volatility
        M : No. of simulations
        b : cost of carry; b = r -----> Default
        
        typ-  "C": Call 
              "P": Put;   typ = C ---> Default
    """
    # Default values
    if typ is None: typ = "C"
    if b is None: b = r
    
    W = np.random.randn(M, 1)
    prices = S0 * np.exp((r - (0.5 * sig **2)) * T + sig * np.sqrt(T) * W)

    p = prices - K
    payoffs = [] 

    
    for i in range(M):
        
        if typ == "C":
            x = p[i] if p[i] > 0 else 0
            payoffs.append(x)
        else:
            # signs reversed!
            x = -p[i] if p[i] < 0 else 0
            payoffs.append(x)
    
    price = sum(payoffs)/M * np.exp(-r * T)

    return price[0]




    
