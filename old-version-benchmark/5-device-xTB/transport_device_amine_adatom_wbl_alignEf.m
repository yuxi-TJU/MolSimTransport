clc
clear

% 从.dat文件中读取数据
h_mm_data = load('amine-hamiltonian.dat');
s_mm_data = load('amine-overlap.dat');
h_mm = 27.2114 * h_mm_data;
s_mm = s_mm_data;

% electrodes
h1_ii_data = load('h1-kpoint-avg.dat');
s1_ii_data = load('s1-kpoint-avg.dat');
h2_ii_data = load('h2-kpoint-avg.dat');
s2_ii_data = load('s2-kpoint-avg.dat');
h1_ii = h1_ii_data(:, 1:2:end) + 1i * h1_ii_data(:, 2:2:end);
s1_ii = s1_ii_data(:, 1:2:end) + 1i * s1_ii_data(:, 2:2:end);
h2_ii = h2_ii_data(:, 1:2:end) + 1i * h2_ii_data(:, 2:2:end);
s2_ii = s2_ii_data(:, 1:2:end) + 1i * s2_ii_data(:, 2:2:end);

N = length(h_mm);
N_lead = length(h1_ii);
efermi = -11.9731;
sum_l = 0;
sum_r = 0;

for j=1:N_lead
    sum_l = sum_l + (h1_ii(j,j) - h_mm(j,j)) / s_mm(j,j);
    sum_r = sum_r + (h2_ii(j,j) - h_mm(N-N_lead+j,N-N_lead+j)) / s_mm(N-N_lead+j,N-N_lead+j);
end
phi = (sum_l + sum_r) / (2*N_lead);

H_mm = h_mm + (phi - efermi) * s_mm;
eigenvalues_align = sort(real(eig(H_mm,s_mm)));

% devide & fold %
Nl = 3 * 16 * 9 + 17 * 9;
Nr = 3 * 16 * 9 + 17 * 9;
Nm = N - Nl - Nr;

H_l = H_mm(1:Nl, 1:Nl);
V_lm = H_mm(1:Nl, Nl+1:Nl+Nm);
V_ml = H_mm(Nl+1:Nl+Nm, 1:Nl);
H_mol = H_mm(Nl+1:Nl+Nm, Nl+1:Nl+Nm);
V_mr = H_mm(Nl+1:Nl+Nm, Nl+Nm+1:N);
V_rm = H_mm(Nl+Nm+1:N, Nl+1:Nl+Nm);
H_r = H_mm(Nl+Nm+1:N, Nl+Nm+1:N);
S_l = s_mm(1:Nl, 1:Nl);
S_lm = s_mm(1:Nl, Nl+1:Nl+Nm);
S_ml = s_mm(Nl+1:Nl+Nm, 1:Nl);
S_mol = s_mm(Nl+1:Nl+Nm, Nl+1:Nl+Nm);
S_mr = s_mm(Nl+1:Nl+Nm, Nl+Nm+1:N);
S_rm = s_mm(Nl+Nm+1:N, Nl+1:Nl+Nm);
S_r = s_mm(Nl+Nm+1:N, Nl+Nm+1:N);


Gamma_s_wba=4;    
Gamma_p_wba=4;    
Gamma_d_wba=4; 
Sigma_s_wba = -0.5j*Gamma_s_wba;
Sigma_p_wba = -0.5j*Gamma_p_wba;
Sigma_d_wba = -0.5j*Gamma_d_wba;
Gamma_gold = [Gamma_d_wba*ones(1,5),Gamma_s_wba,Gamma_p_wba*ones(1,3)];
Sigma_gold = [Sigma_d_wba*ones(1,5),Sigma_s_wba,Sigma_p_wba*ones(1,3)];
Gamma_layer_3 = [Gamma_gold,Gamma_gold,Gamma_gold,Gamma_gold];
Gamma_layer = [Gamma_layer_3,Gamma_layer_3,Gamma_layer_3,Gamma_layer_3];
Sigma_layer_3 = [Sigma_gold,Sigma_gold,Sigma_gold,Sigma_gold];
Sigma_layer = [Sigma_layer_3,Sigma_layer_3,Sigma_layer_3,Sigma_layer_3];
Sigma_3layer_L = diag([Sigma_layer,Sigma_layer,Sigma_layer,zeros(1,17*9)]);
Sigma_3layer_R = diag([zeros(1,17*9),Sigma_layer,Sigma_layer,Sigma_layer]);

% 构建与导线连接的矩阵
E_num = 401;  
  
Erange = linspace(-2, 2, E_num);  
dE = abs(Erange(2) - Erange(1));
nbf = size(h_mm, 1);
z_plus = 1e-13j;

eigenvalues_mol = sort(real(eig(H_mol,S_mol)));

[C, E] = eig(H_mol,S_mol);
Sigma_dft = diag([zeros(1,33),0.5,0.5,zeros(1,Nm-35)]);
E_shift = E + Sigma_dft;

H_shift = C * E_shift * inv(C); 



for i=1:length(Erange)       

    Sigma_L = (Erange(i) * S_ml - V_ml) * inv(Erange(i) * S_l - H_l - Sigma_3layer_L) * (Erange(i) * S_lm - V_lm);
    Sigma_R = (Erange(i) * S_mr - V_mr) * inv(Erange(i) * S_r - H_r - Sigma_3layer_R) * (Erange(i) * S_rm - V_rm);
    Gamma_L = 1j * (Sigma_L - Sigma_L');
    Gamma_R = 1j * (Sigma_R - Sigma_R');
    
 
    G = inv((Erange(i) + z_plus) * eye(Nm) - H_shift - Sigma_L - Sigma_R);   
    T_list= Gamma_L * G * Gamma_R * G';
    T(i) = real(trace(T_list));
end

figure(2)
plot(Erange,log10(T),'-','linewidth',1.5);hold on
title('Transmission')
xlim([-2, 2]);
xlabel('E(eV)')
ylabel('Transmission (log10)')


