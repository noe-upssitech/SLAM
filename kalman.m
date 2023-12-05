[N, T, M, Z, arrayH, arrayR, in_fov, F, B, CC, Hfull, mX0, PX0, Qw, Rv, X] = dataSimulation(0)

% update loop
% Initialisation du filtre de Kalman
mX_est = mX0;
PX_est = PX0;

% Matrices du filtre de Kalman
H_kalman = Hfull;
R_kalman = Rv;

% Préallocation pour les états estimés
X_estimated = nan(size(mX0, 1), N);

% Boucle principale de simulation et mise à jour du filtre de Kalman
for k = 2:N
    % Utilisez le filtre de Kalman pour estimer l'état du système
    mX_pred = F * mX_est + B;
    PX_pred = F * PX_est * F' + Qw;

    % Mise à jour de l'état avec les mesures
    if any(~isnan(Z(:, k)))
        % Mesures disponibles, utilisez le filtre de Kalman
        K = PX_pred * H_kalman' / (H_kalman * PX_pred * H_kalman' + R_kalman);
        mX_est = mX_pred + K * (Z(:, k) - H_kalman * mX_pred);
        PX_est = (eye(size(PX0)) - K * H_kalman) * PX_pred;
    else
        % Aucune mesure disponible, utilisez uniquement le modèle dynamique
        mX_est = mX_pred;
        PX_est = PX_pred;
    end

    % Enregistrement des résultats du filtre de Kalman
    X_estimated(:, k) = mX_est;
    
end

X_estimated