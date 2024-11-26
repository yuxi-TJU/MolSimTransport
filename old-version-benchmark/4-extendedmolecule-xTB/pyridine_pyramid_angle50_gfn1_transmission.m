clear
clc

Hmatrix = readmatrix('pyridine-pyramid-angle50-gfn1_hamiltonian_matrix_xtb.csv');
H_without_first_column = Hmatrix(2:end,: );
H = 27.2114 * H_without_first_column;

Smatrix = readmatrix('pyridine-pyramid-angle50-gfn1_overlap_matrix_xtb.csv');
S = Smatrix(2:end,: ); 

Cmatrix = readmatrix('pyridine-pyramid-angle50-gfn1_orbital-coefficients_xtb.csv');
C_without_first_column = Cmatrix(2:end,: );
C = C_without_first_column';

Ematrix = diag(readmatrix('pyridine-pyramid-angle50-gfn1_orbital-energies_xtb.csv'));
E = Ematrix(2:end, 2:end);


N = length(H);  
EEEE=linspace(0.5,0.5,N);
E_num = 2001;  
Erange = linspace(-12,-9, E_num);  
dE = abs(Erange(2) - Erange(1));  

% full hamiltonian 
H_scc = C * E * inv(C);

[eigenvector2,eigenvalue2]=eig(H_scc);
eigenvalue2o=eigenvalue2;
eigenvalue2=sort(diag(eigenvalue2));

z_plus = 1e-13j;

Gamma_s_wba=4;    
Gamma_p_wba=4;    
Gamma_d_wba=4;  
Sigma_s_wba = -0.5j*Gamma_s_wba;
Sigma_p_wba = -0.5j*Gamma_p_wba;
Sigma_d_wba = -0.5j*Gamma_d_wba;
Gamma_Au = [Gamma_d_wba*ones(1,5),Gamma_s_wba,Gamma_p_wba*ones(1,3)];
Sigma_Au = [Sigma_d_wba*ones(1,5),Sigma_s_wba,Sigma_p_wba*ones(1,3)];

Gamma_matrix = [Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au,Gamma_Au];
Sigma_matrix = [Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au,Sigma_Au];

for i=1:length(Erange)        
    Gamma1 = diag([Gamma_matrix,zeros(1,N-9*10)]);
    Gamma2 = diag([zeros(1,N-9*10),Gamma_matrix]);
    Sigma1 = diag([Sigma_matrix,zeros(1,N-9*10)]);
    Sigma2 = diag([zeros(1,N-9*10),Sigma_matrix]);

    G = inv(((Erange(i) + z_plus) * eye(N)) - H_scc - Sigma1 - Sigma2);  
          
    T_list= Gamma1 * G * Gamma2 * G';
    T(i) = real(trace(T_list));
end


figure(1)
plot(Erange,log10(T),'-','linewidth',1.5);hold on
scatter(eigenvalue2,EEEE,'MarkerEdgeColor','b','LineWidth',1,'SizeData',50);
xlim([-12,-9]);
xlabel('E(eV)')
ylabel('Transmission (log10)')
