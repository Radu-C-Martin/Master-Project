classdef gpCallback < casadi.Callback
  properties
    model
  end
  methods
    function self = gpCallback(name)
      self@casadi.Callback();
      construct(self, name, struct('enable_fd', true));
    end

    % Number of inputs and outputs
    function v=get_n_in(self)
      v=1;
    end
    function v=get_n_out(self)
      v=1;
    end
    % Function sparsity
    function v=get_sparsity_in(self, i)
      v=casadi.Sparsity.dense(7, 1);
    end

    % Initialize the object
    function init(self)
      disp('initializing gpCallback')
      gpr = load('gpr_model.mat', 'model');
      self.model = gpr.model;
    end

    % Evaluate numerically
    function arg = eval(self, arg)
      x = full(arg{1});
      % Transpose x since gp predictor takes row by row, and casadi gives
      % colum by column
      [mean, ~] = predict(self.model, x');
      arg = {mean};
    end
  end
end