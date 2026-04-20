"""
Deep Q-Network (DQN) model for Personalized HRI with LSTM-based late fusion.

This module defines the recurrent policy network used in the paper's RL framework.
The model processes a multimodal state representation and predicts Q-values for
all possible robot actions.

Architecture overview:
- Each input modality is first projected to its own embedding space
- Embeddings are concatenated through a late-fusion strategy
- The fused representation is processed by an LSTM for temporal modeling
- A final MLP outputs Q-values for the available robot actions

Supported modalities (depending on configuration):
- Action prediction features
- Action recognition features
- Visual working memory / task-related features
- Objects-in-table (OiT) state
- ASR signals
- Optional temporal-context features
"""

import torch
import torch.nn as nn

class DQN_LSTM_LateFusion(nn.Module):
    """
    Recurrent DQN with modality-specific embeddings and late fusion.
    
    The model is designed for sequential decision making in HRI settings, where
    state observations may include multiple heterogeneous modalities. Each
    modality is embedded independently, then concatenated and passed through an
    LSTM to capture temporal dependencies across interaction steps.
    """
    def __init__(self, input_size, output_size, cfg):
        super(DQN_LSTM_LateFusion, self).__init__()

        self.cfg = cfg 
        self.input_size = input_size
        self.hidden_size_LSTM = self.cfg.HIDDEN_SIZE_LSTM
        self.output_size = output_size
        
        
        # Number of modality embeddings that will be concatenated later.
        # Starts with VWM + OiT, and grows depending on enabled modalities.
        num_embedding = 2
        
        # Set default hyperparameters if not explicitly provided in the config.
        if not hasattr(self.cfg, 'EMBEDDING_SIZE'):  # returns True
            self.cfg.EMBEDDING_SIZE = 32
        
        if not hasattr(self.cfg, 'GLOBAL_EMBEDDING_SIZE'):  # returns True
            self.cfg.GLOBAL_EMBEDDING_SIZE = 256
        
        if not hasattr(self.cfg, 'MLP_SIZE'):  # returns True
            self.cfg.MLP_SIZE = 256
            
        if hasattr(self.cfg, 'NOISE_STD'):  # returns True
            self.noise_std = self.cfg.NOISE_STD
        else:
            self.noise_std = 0
            
        self.embedding_size = self.cfg.EMBEDDING_SIZE
        self.global_embedding_size = self.cfg.GLOBAL_EMBEDDING_SIZE
        self.mlp_size = self.cfg.MLP_SIZE
        # Initialize hidden states for recurrent execution.
        self.resetHidden()

        # Modality-specific input projections.
        # Each available modality is independently embedded before fusion.
        if self.cfg.NOT_PRED == False:
            num_embedding += 1
            self.input_AcPred = nn.Sequential(
                nn.Linear(self.cfg.FIDX[0], self.embedding_size),
                torch.nn.LayerNorm(self.embedding_size),
                nn.ReLU()
            )
        if self.cfg.NOT_REC == False:
            num_embedding += 1
            self.input_AcRec = nn.Sequential(
                nn.Linear(self.cfg.FIDX[1]-self.cfg.FIDX[0], self.embedding_size),
                torch.nn.LayerNorm(self.embedding_size),
                nn.ReLU()            
            )
        
        self.input_VWM = nn.Sequential(
            nn.Linear(self.cfg.FIDX[2]-self.cfg.FIDX[1], self.embedding_size),
            torch.nn.LayerNorm(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_OiT = nn.Sequential(
            nn.Linear(self.cfg.FIDX[3]-self.cfg.FIDX[2], self.embedding_size),
            torch.nn.LayerNorm(self.embedding_size),
            nn.ReLU()            
        )
        
        if self.cfg.NOT_ASR == False:
            num_embedding += 1
            self.input_ASR = nn.Sequential(
                nn.Linear(self.cfg.FIDX[4]-self.cfg.FIDX[3], self.embedding_size),
                torch.nn.LayerNorm(self.embedding_size),
                nn.ReLU()            
            )
       
        # Global fusion layer applied after concatenating all modality embeddings.
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_size*num_embedding, self.global_embedding_size ),
            torch.nn.LayerNorm(self.global_embedding_size),
            nn.ReLU()
        )

        # Temporal model over the fused embeddings
        self.lstm = nn.LSTM(self.global_embedding_size, self.hidden_size_LSTM, batch_first = True)
        self.norm = torch.nn.LayerNorm(self.hidden_size_LSTM)
        
        # Final MLP that maps the temporal representation to Q-values.
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size_LSTM, self.mlp_size),
            nn.ReLU(),
            nn.Linear(self.mlp_size, output_size)
        )       
       
        
    def resetHidden(self):
        """
        Reset the internal LSTM hidden state.

        This is typically called at the beginning of a new episode.
        Both the current hidden state and the previously stored hidden state
        are reset to zeros.
        """
        self.hidden = (torch.zeros(1,self.hidden_size_LSTM).to(self.cfg.DEVICE),
                       torch.zeros(1,self.hidden_size_LSTM).to(self.cfg.DEVICE))
        self.prevHidden = (torch.zeros(1,self.hidden_size_LSTM).to(self.cfg.DEVICE),
                           torch.zeros(1,self.hidden_size_LSTM).to(self.cfg.DEVICE))
        return
    
    def init_lstm_weights(self):
        """
        Initialize LSTM weights.

        - Input weights use Xavier initialization
        - Recurrent weights use orthogonal initialization
        - Biases are initialized to zero
        """
        for name, param in self.lstm.named_parameters():
           if "weight_ih" in name:
               for sub_name in ["weight_ih_i", "weight_ih_f", "weight_ih_g", "weight_ih_o"]:
                   if sub_name in name:
                       nn.init.xavier_uniform_(param.data)
           elif "weight_hh" in name:
               for sub_name in ["weight_hh_i", "weight_hh_f", "weight_hh_g", "weight_hh_o"]:
                   if sub_name in name:
                       nn.init.orthogonal_(param.data)
           elif "bias" in name:
               nn.init.constant_(param.data, 0)
                
               
    def init_weights(self):
        """
        Initialize linear and convolutional layers with Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def init_zero_weights(self):
        """
        Initialize the last output layer to zeros.

        This can be useful at the beginning of training to control the initial
        action-value estimates.
        """
        nn.init.constant_(self.output_layer[-1].weight, 0)
        nn.init.constant_(self.output_layer[-1].bias, 0)
  
        
    def forward(self, inputs, hidden=None):
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input state tensor. It can be either:
            - [batch, features]
            - [batch, time, features]
        hidden : tuple or None
            Optional LSTM hidden state. If None, the internally stored hidden
            state is used.

        Returns
        -------
        tuple
            - Q-values for each action
            - updated LSTM hidden state
        """
        # Use provided hidden state if available; otherwise use internal state.
        hiddenT = self.hidden if hidden is None else hidden
        
        # Split the input state into its modality-specific components.
        # The exact slicing depends on which modalities are enabled/disabled.
        if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
            vwm = inputs[...,self.cfg.FIDX[1]-64:self.cfg.FIDX[2]-64]
            oit = inputs[...,self.cfg.FIDX[2]-64:self.cfg.FIDX[3]-64]
            if self.cfg.NOT_ASR == False:
                asr = inputs[...,self.cfg.FIDX[3]-64:self.cfg.FIDX[4]-64]
      
        elif self.cfg.NOT_PRED:
            ac_rec = inputs[...,self.cfg.FIDX[0]-32:self.cfg.FIDX[1]-32]
            vwm = inputs[...,self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]
            oit = inputs[...,self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]
            if self.cfg.NOT_ASR == False:
                asr = inputs[...,self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32]
        elif self.cfg.NOT_REC:
            ac_pred = inputs[..., :self.cfg.FIDX[0]]
            vwm = inputs[...,self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]
            oit = inputs[...,self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]
            if self.cfg.NOT_ASR == False:
                asr = inputs[...,self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32]
        else:
            ac_pred = inputs[..., :self.cfg.FIDX[0]]
            ac_rec = inputs[...,self.cfg.FIDX[0]:self.cfg.FIDX[1]]
            vwm = inputs[...,self.cfg.FIDX[1]:self.cfg.FIDX[2]]
            oit = inputs[...,self.cfg.FIDX[2]:self.cfg.FIDX[3]]
            if self.cfg.NOT_ASR == False:
                asr = inputs[...,self.cfg.FIDX[3]:self.cfg.FIDX[4]]
            
        # Compute modality-specific embeddings.
        if self.cfg.NOT_PRED == False:
            x1 = self.input_AcPred(ac_pred)
        if self.cfg.NOT_REC == False:
            x2 = self.input_AcRec(ac_rec)
        x3 = self.input_VWM(vwm)
        x4 = self.input_OiT(oit)
        if self.cfg.NOT_ASR == False:
            
            x5 = self.input_ASR(asr)
        
        # Concatenate available embeddings using late fusion.
        if self.cfg.NOT_ASR == False:
            
            if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
                x = torch.cat((  x3, x4,x5), len(inputs.shape)-1)  
            
            elif self.cfg.NOT_REC:
                x = torch.cat(( x1, x3, x4,x5), len(inputs.shape)-1) 
            elif self.cfg.NOT_PRED:
                 x = torch.cat(( x2, x3, x4,x5), len(inputs.shape)-1) 
            else:
                x = torch.cat(( x1, x2, x3, x4,x5), len(inputs.shape)-1) 
        else:
            if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
                x = torch.cat((  x3, x4), len(inputs.shape)-1)  
            
            elif self.cfg.NOT_REC:
                x = torch.cat(( x1, x3, x4), len(inputs.shape)-1) 
            elif self.cfg.NOT_PRED:
                x = torch.cat(( x2, x3, x4), len(inputs.shape)-1) 
            else:
                x = torch.cat(( x1, x2, x3, x4), len(inputs.shape)-1) 
        
        # Project fused embedding to the global latent space.
        x = self.hidden1(x)
        # Temporal modeling across interaction steps.
        output, (hx, cx) = self.lstm(x, hiddenT)
        output = self.norm(output)
        # Optional output noise during training for regularization.
        if self.noise_std>0 and self.training:
            noise_level = self.noise_std*output.std()
            output = output + noise_level*torch.ones_like(output)
            
        # Final Q-values for all robot actions.
        x = self.output_layer(output)

        # If the hidden state was not externally provided, update the internal
        # recurrent state used for sequential inference.
        if hidden is None:
            self.prevHidden = (self.hidden[0].clone(), self.hidden[1].clone())
            self.hidden = (hx,cx)

        return x, (hx,cx) 










