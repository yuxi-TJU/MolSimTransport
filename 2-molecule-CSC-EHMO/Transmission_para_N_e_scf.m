clear
clc

Hmatrix = readmatrix('para-N-e-scf-hamiltonian.csv');
H = 27.2114 * Hmatrix(2:end,: );
Smatrix = readmatrix('para-N-e-scf-overlap.csv');
S = Smatrix(2:end,: ); 


% 求分子轨道能量与波函数
[eigen_vector,eigen_value]=eig(H,S);
eigenvalue=diag(eigen_value);


N = length(H);  
No = length(eigenvalue);
EEEE=linspace(0.5,0.5,No);
E_num = 2000;  
Erange = linspace(-15, -5, E_num);  
dE = abs(Erange(2) - Erange(1));  


Gamma_s_wba=0.2;    
Gamma_p_wba=0.2;   
Gamma_d_wba=0.2; 
Sigma_s_wba = -0.5j*Gamma_s_wba;
Sigma_p_wba = -0.5j*Gamma_p_wba;
Sigma_d_wba = -0.5j*Gamma_d_wba;
z_plus = 1e-13j;

%对位
for i=1:length(Erange)        
Gamma1 = diag([0,Gamma_s_wba,Gamma_p_wba*ones(1,3),zeros(1,N-5)]);
Gamma2 = diag([zeros(1,N-5),Gamma_s_wba,Gamma_p_wba*ones(1,3),0]);
Sigma1 = diag([0,Sigma_s_wba,Sigma_p_wba*ones(1,3),zeros(1,N-5)]);
Sigma2 = diag([zeros(1,N-5),Sigma_s_wba,Sigma_p_wba*ones(1,3),0]);


     G = inv(((Erange(i) + z_plus) * S) - H - Sigma1 - Sigma2);  
          
    Tpara_list= Gamma1 * G * Gamma2 * G';
    Tpara(i) = real(trace(Tpara_list));
end


figure(2)
plot(Erange,log10(Tpara),'-','linewidth',1.5);hold on
scatter(eigenvalue,EEEE,'MarkerEdgeColor','b','LineWidth',1,'SizeData',50);
title('Transmission (N_e_scf,WBA)')
xlim([-10 -7]);
xlabel('E(eV)')
ylabel('Transmission (log10)')
