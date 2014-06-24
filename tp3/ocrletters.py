import numpy as np

A = np.array([
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1,
-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1])
B = np.array([
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1])
C = np.array([
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1])
D = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1])
E = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
F = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
G = np.array([
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1])
H = np.array([
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1])
I = np.array([
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1])
J = np.array([
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
K = np.array([
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1])
L = np.array([
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1])
M = np.array([
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,
 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
 1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1,
 1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1])
N = np.array([
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1])
O = np.array([
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1])
P = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
Q = np.array([
-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1])
R = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1])
S = np.array([
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1])
T = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1])
U = np.array([
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,
-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1])
V = np.array([
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1])
W = np.array([
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,
-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1,
-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1])
X = np.array([
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
-1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,
-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
-1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,
-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,
-1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1])
Y = np.array([
 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1,
-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1])
Z = np.array([
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,
-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,
-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,
-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])