clear
clc

Hmatrix = readmatrix('AQ-gfn1_hamiltonian_matrix_xtb.csv');
H_without_first_column = Hmatrix(2:end,: );
H = 27.2114 * H_without_first_column;

Smatrix = readmatrix('AQ-gfn1_overlap_matrix_xtb.csv');
S = Smatrix(2:end,: ); 

Cmatrix = readmatrix('AQ-gfn1_orbital-coefficients_xtb.csv');
C_without_first_column = Cmatrix(2:end,: );
C = C_without_first_column';

Ematrix = diag(readmatrix('AQ-gfn1_orbital-energies_xtb.csv'));
E = Ematrix(2:end, 2:end);


N = length(H);  
EEEE=linspace(0.5,0.5,N);
E_num = 2000;  
Erange = linspace(-15,-5, E_num);  
dE = abs(Erange(2) - Erange(1));  

% full hamiltonian 
H_scc = C * E * inv(C);

[eigenvector2,eigenvalue2]=eig(H_scc);
eigenvalue2o=eigenvalue2;
eigenvalue2=sort(diag(eigenvalue2));

z_plus = 1e-13j;

Gamma_s_wba=0.2;    
Gamma_p_wba=0.2;   
Gamma_d_wba=0.2;  
Sigma_s_wba = -0.5j*Gamma_s_wba;
Sigma_p_wba = -0.5j*Gamma_p_wba;
Sigma_d_wba = -0.5j*Gamma_d_wba;



for i=1:length(Erange)        
    Gamma1 = diag([Gamma_s_wba,Gamma_p_wba*ones(1,3),Gamma_d_wba*ones(1,5),zeros(1,N-9)]);
    Gamma2 = diag([zeros(1,9),Gamma_s_wba,Gamma_p_wba*ones(1,3),Gamma_d_wba*ones(1,5),zeros(1,N-18)]);
    Sigma1 = diag([Sigma_s_wba,Sigma_p_wba*ones(1,3),Sigma_d_wba*ones(1,5),zeros(1,N-9)]);
    Sigma2 = diag([zeros(1,9),Sigma_s_wba,Sigma_p_wba*ones(1,3),Sigma_d_wba*ones(1,5),zeros(1,N-18)]);

    G = inv(((Erange(i) + z_plus) * eye(N)) - H_scc - Sigma1 - Sigma2);  
          
    T_list= Gamma1 * G * Gamma2 * G';
    T(i) = real(trace(T_list));
end

figure;
plot(Erange,log10(T),'-','linewidth',1.5);hold on
scatter(eigenvalue2,EEEE,'MarkerEdgeColor','b','LineWidth',1,'SizeData',50);
xlim([-12,-8]);
xlabel('E(eV)')
ylabel('Transmission (log10)')
