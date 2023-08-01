# -*- coding: utf-8 -*-
"""SFP.py
"""
import pandas as pd
import numpy as np
import math
import ta
import matplotlib.pyplot as plt

class SFP:
    '''
    Class of Swing Failure Pattern by EmreKb
    '''
    def __init__(self, df, swing_threshold, lookback = 4899, is_opposite = False, Pivot_length = 21):
      self.df = df.copy()
      self.swing_threshold = swing_threshold
      self.lookback = lookback
      self.is_opposite = is_opposite
      self.pivot_length = Pivot_length
    
    def _get_candle(self, index):
        return self.df['Open'][index], self.df['High'][index], self.df['Low'][index], self.df['Close'][index], index
    
    def _sfp(self):
      # Function to calculate the Swing Failure Patten
      so, sh, sl, sc, si = self._f_get_candle(0)

      ph = ta.pivothigh(self.pivot_length, 0)
      pl = ta.pivotlow(self.pivot_length, 0)
      
      # High SFP
      hc1 = ph
      maxp = self.df['High'][1]
      hc2 = False
      hx = 0
      hy = 0.0
      
      for i in range(1, self.lookback + 1):
          co, ch, cl, cc, ci = self._get_candle(i)
          if ch >= sh:
              break
          if ch < sh and ch > max(so, sc) and ph[self.df['bar_index'][i] - ci] and ch > maxp:
              hc2 = True
              hx = self.df['bar_index'][i] 
              hy = ch
          if ch > maxp:
              maxp = ch
        
      hcs = hc1 & hc2
      
      # Low SFP
      lc1 = pl
      minp = self.df['Low'][1]
      lc2 = False
      lx = 0
      ly = 0.0
      for i in range(2, self.lookback + 1):
        co, ch, cl, cc, ci = self._get_candle(i)
        if cl < sl:
            break
        if sl < cl and min(so, sc) > cl and pl[self.df['bar_index'][i] - ci] and cl < minp:
            lc2 = True
            lx = self.df['bar_index'][i] 
            ly = cl
        if cl < minp:
            minp = ch
        
        lcs = lc1 & lc2
        
        return hcs, hx, hy, lcs, lx, ly
    
    def _calculate_sfp(self):
      # Calculate SFP and assign the results to dataframe columns
      hsfp, hx, hy, lsfp, lx, ly = self._sfp()
      if self.is_opposite:
        hsfp = hsfp if (hsfp & self.df['Open'] > self.df['Close']) else none
        lsfp = lsfp if (lsfp & self.df['Open'] < self.df['Close']) else none

      SFP_Zero_Lag = hsfp - lsfp
      self.df['sfp'] = SFP_Zero_Lag
    	
    def get_data(self):
      return self.df