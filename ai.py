from typing import List, Tuple, Dict, Optional
import random
import math


class AdversarialSearchAI:
    """PROPER adversarial search implementation with recursive min/max functions"""
    
    def __init__(self):
        self.payoff_matrix = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        self.search_depth = 3
        
    def minimax(self, depth: int, is_maximizing: bool, alpha: float, beta: float,
                my_history: List[str], opp_history: List[str],
                my_score: int, opp_score: int) -> Tuple[float, Optional[str]]:
        """
        PROPER recursive minimax with alpha-beta pruning
        Uses alternating min and max nodes
        """
        # Base case: leaf node or depth limit
        if depth == 0 or len(my_history) >= 20:
            return self.evaluate_state(my_history, opp_history, my_score, opp_score, is_maximizing), None
        
        best_move = None
        
        if is_maximizing:
            # MAXIMIZING PLAYER (our AI)
            max_eval = float('-inf')
            
            for move in ['C', 'D']:
                # For each possible move, consider opponent's best responses
                for opp_move in ['C', 'D']:
                    # Calculate new scores
                    new_my_score = my_score + self.payoff_matrix[(move, opp_move)][0]
                    new_opp_score = opp_score + self.payoff_matrix[(move, opp_move)][1]
                    
                    # Create new history
                    new_my_history = my_history + [move]
                    new_opp_history = opp_history + [opp_move]
                    
                    # RECURSIVE CALL to MINIMIZING player
                    eval_score, _ = self.minimax(
                        depth - 1, False, alpha, beta,
                        new_my_history, new_opp_history,
                        new_my_score, new_opp_score
                    )
                    
                    # Update max evaluation
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    
                    # Alpha-beta pruning
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_move
            
        else:
            # MINIMIZING PLAYER (opponent)
            min_eval = float('inf')
            
            for move in ['C', 'D']:
                # For each opponent move, consider our best responses
                for our_move in ['C', 'D']:
                    # Calculate new scores (from opponent's perspective)
                    new_my_score = my_score + self.payoff_matrix[(our_move, move)][0]
                    new_opp_score = opp_score + self.payoff_matrix[(our_move, move)][1]
                    
                    # Create new history
                    new_my_history = my_history + [our_move]
                    new_opp_history = opp_history + [move]
                    
                    # RECURSIVE CALL to MAXIMIZING player
                    eval_score, _ = self.minimax(
                        depth - 1, True, alpha, beta,
                        new_my_history, new_opp_history,
                        new_my_score, new_opp_score
                    )
                    
                    # Update min evaluation (opponent minimizes our score)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    
                    # Alpha-beta pruning
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_move
    
    def evaluate_state(self, my_history: List[str], opp_history: List[str], 
                      my_score: int, opp_score: int, is_maximizing: bool) -> float:
        """Evaluate game state from current player's perspective"""
        if not my_history:
            return 0.0
        
        # Base score difference
        score_diff = my_score - opp_score
        
        # Strategic evaluation
        strategic_value = 0.0
        
        # Cooperation patterns
        if len(my_history) >= 3:
            my_recent_coop = my_history[-3:].count('C') / 3.0
            opp_recent_coop = opp_history[-3:].count('C') / 3.0 if len(opp_history) >= 3 else 0.5
            
            # Reward mutual cooperation potential
            cooperation_bonus = min(my_recent_coop, opp_recent_coop) * 2.0
            strategic_value += cooperation_bonus
            
            # Penalize exploitation
            if my_recent_coop > opp_recent_coop + 0.4:
                strategic_value -= 3.0  # We're being too nice to a defector
        
        # Pattern unpredictability
        if len(my_history) >= 5:
            entropy = self.calculate_entropy(my_history)
            strategic_value += entropy * 1.5
        
        # If we're evaluating for minimizing player (opponent), invert the perspective
        if not is_maximizing:
            return -score_diff + strategic_value
        else:
            return score_diff + strategic_value
    
    def calculate_entropy(self, history: List[str]) -> float:
        """Calculate move pattern entropy"""
        if len(history) < 2:
            return 0.5
            
        transitions = []
        for i in range(1, len(history)):
            transitions.append(history[i-1] + history[i])
            
        unique_transitions = len(set(transitions))
        max_possible = min(4, len(transitions))
        
        return unique_transitions / max_possible if max_possible > 0 else 0
    
    def get_best_move(self, my_history: List[str], opp_history: List[str], 
                     my_score: int, opp_score: int) -> str:
        """Public method to get best move using minimax"""
        if len(my_history) < 2:
            # Use simple heuristic for early game
            if not opp_history:
                return 'C'
            coop_rate = opp_history.count('C') / len(opp_history)
            return 'C' if coop_rate > 0.6 else 'D'
        
        # Start minimax search as maximizing player
        current_score = self.calculate_current_score(my_history, opp_history)
        current_opp_score = self.calculate_opponent_score(my_history, opp_history)
        
        _, best_move = self.minimax(
            depth=self.search_depth,
            is_maximizing=True,  # We are the maximizing player
            alpha=float('-inf'),
            beta=float('inf'),
            my_history=my_history,
            opp_history=opp_history,
            my_score=current_score,
            opp_score=current_opp_score
        )
        
        return best_move if best_move else 'C'
    
    def calculate_current_score(self, my_history: List[str], opp_history: List[str]) -> int:
        """Calculate current total score for our player"""
        total = 0
        for i in range(len(my_history)):
            my_move = my_history[i]
            opp_move = opp_history[i] if i < len(opp_history) else 'C'
            total += self.payoff_matrix[(my_move, opp_move)][0]
        return total
    
    def calculate_opponent_score(self, my_history: List[str], opp_history: List[str]) -> int:
        """Calculate current total score for opponent"""
        total = 0
        for i in range(len(opp_history)):
            my_move = my_history[i] if i < len(my_history) else 'C'
            opp_move = opp_history[i]
            total += self.payoff_matrix[(my_move, opp_move)][1]
        return total


# Keep the existing FuzzyLogicSystem and AdvancedStrategyAnalyzer classes exactly as they were
class FuzzyLogicSystem:
    """Fuzzy logic system for decision making"""
    def __init__(self):
        # Fuzzy sets for cooperation rate
        self.coop_low = (0.0, 0.0, 0.3)
        self.coop_medium = (0.2, 0.5, 0.8)
        self.coop_high = (0.7, 1.0, 1.0)
        
        # Fuzzy sets for pattern consistency
        self.consistency_low = (0.0, 0.0, 0.4)
        self.consistency_medium = (0.3, 0.5, 0.7)
        self.consistency_high = (0.6, 1.0, 1.0)
        
        # Output fuzzy sets for cooperation tendency
        self.tendency_defect = (0.0, 0.0, 0.3)
        self.tendency_balanced = (0.2, 0.5, 0.8)
        self.tendency_cooperate = (0.7, 1.0, 1.0)
    
    def triangular_mf(self, x, triangle):
        """Triangular membership function"""
        a, b, c = triangle
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def fuzzy_inference(self, coop_rate: float, consistency: float, score_diff: int) -> float:
        """Perform fuzzy inference to determine cooperation tendency"""
        # Fuzzify inputs
        coop_low_val = self.triangular_mf(coop_rate, self.coop_low)
        coop_medium_val = self.triangular_mf(coop_rate, self.coop_medium)
        coop_high_val = self.triangular_mf(coop_rate, self.coop_high)
        
        consistency_low_val = self.triangular_mf(consistency, self.consistency_low)
        consistency_medium_val = self.triangular_mf(consistency, self.consistency_medium)
        consistency_high_val = self.triangular_mf(consistency, self.consistency_high)
        
        # Score difference fuzzy sets (simplified)
        if score_diff < -10:
            score_losing_val = 1.0
            score_even_val = 0.0
            score_winning_val = 0.0
        elif -10 <= score_diff <= 10:
            score_losing_val = 0.0
            score_even_val = 1.0
            score_winning_val = 0.0
        else:
            score_losing_val = 0.0
            score_even_val = 0.0
            score_winning_val = 1.0
        
        # Fuzzy rules
        rules = []
        
        # Rule 1: If cooperation is high and consistent, tend to cooperate
        if coop_high_val > 0 and consistency_high_val > 0:
            strength = min(coop_high_val, consistency_high_val)
            rules.append((strength, self.tendency_cooperate))
        
        # Rule 2: If cooperation is low and consistent, tend to defect
        if coop_low_val > 0 and consistency_high_val > 0:
            strength = min(coop_low_val, consistency_high_val)
            rules.append((strength, self.tendency_defect))
        
        # Rule 3: If inconsistent pattern, balanced approach
        if consistency_low_val > 0:
            strength = consistency_low_val
            rules.append((strength, self.tendency_balanced))
        
        # Rule 4: If winning significantly, can afford to cooperate
        if score_winning_val > 0 and coop_medium_val > 0:
            strength = min(score_winning_val, coop_medium_val)
            rules.append((strength, self.tendency_cooperate))
        
        # Rule 5: If losing significantly, might need to defect
        if score_losing_val > 0 and coop_low_val > 0:
            strength = min(score_losing_val, coop_low_val)
            rules.append((strength, self.tendency_defect))
        
        # Rule 6: Default balanced approach
        if coop_medium_val > 0 and consistency_medium_val > 0:
            strength = min(coop_medium_val, consistency_medium_val)
            rules.append((strength, self.tendency_balanced))
        
        # Defuzzify using centroid method
        if not rules:
            return 0.5  # Neutral tendency
        
        numerator = 0.0
        denominator = 0.0
        
        for strength, output_set in rules:
            # Use the centroid of the output fuzzy set weighted by rule strength
            centroid = (output_set[0] + output_set[1] + output_set[2]) / 3
            numerator += strength * centroid
            denominator += strength
        
        if denominator == 0:
            return 0.5
        
        return max(0.0, min(1.0, numerator / denominator))


class AdvancedStrategyAnalyzer:
    """Advanced strategy analyzer with machine learning principles"""
    def __init__(self):
        self.available_strategies = ["minimax", "fuzzy", "tit_for_tat", "adaptive", "pattern_matcher", "bayesian"]
        self.current_strategy = "minimax"
        self.opponent_pattern = []
        self.strategy_performance = {s: {"wins": 0, "uses": 0} for s in self.available_strategies}
        self.analysis_accuracy = 0.85
        self.pattern_memory = 15
        self.confidence_threshold = 0.7
        
    def analyze_opponent(self, opp_history: List[str], my_score: int, opp_score: int) -> str:
        if len(opp_history) < 3:
            return "minimax"
            
        analysis_confidence = self.calculate_analysis_confidence(opp_history)
        
        if random.random() > self.analysis_accuracy * analysis_confidence:
            return self.get_weighted_random_strategy()
            
        coop_rate = opp_history.count('C') / len(opp_history)
        recent_coop = opp_history[-3:].count('C') / min(3, len(opp_history))
        pattern_consistency = self.calculate_pattern_consistency(opp_history)
        score_differential = my_score - opp_score
        
        strategy_scores = {}
        
        # Strategy selection based on opponent behavior
        if coop_rate > 0.8 and pattern_consistency > 0.7:
            # Cooperative and predictable opponent
            strategy_scores["pattern_matcher"] = 0.9
            strategy_scores["fuzzy"] = 0.7
        elif coop_rate < 0.2 and pattern_consistency > 0.6:
            # Defective and predictable opponent  
            strategy_scores["minimax"] = 0.9
            strategy_scores["adaptive"] = 0.6
        elif pattern_consistency < 0.4:
            # Unpredictable opponent
            strategy_scores["fuzzy"] = 0.8
            strategy_scores["adaptive"] = 0.7
        elif abs(coop_rate - 0.5) < 0.2 and score_differential < -5:
            # Balanced opponent but we're losing
            strategy_scores["minimax"] = 0.8
            strategy_scores["bayesian"] = 0.6
        else:
            # Default strategies
            strategy_scores["fuzzy"] = 0.7
            strategy_scores["adaptive"] = 0.6
            
        # Adjust scores based on historical performance
        for strategy in strategy_scores:
            success_rate = self.strategy_performance[strategy]["wins"] / max(1, self.strategy_performance[strategy]["uses"])
            strategy_scores[strategy] *= (0.5 + success_rate)
            
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        return best_strategy
        
    def calculate_analysis_confidence(self, opp_history: List[str]) -> float:
        if len(opp_history) < 5:
            return 0.5
        consistency = self.calculate_pattern_consistency(opp_history)
        data_quality = min(1.0, len(opp_history) / 20)
        return (consistency + data_quality) / 2
        
    def calculate_pattern_consistency(self, opp_history: List[str]) -> float:
        if len(opp_history) < 3:
            return 0.5
        transitions = []
        for i in range(1, len(opp_history)):
            transitions.append(opp_history[i-1] + opp_history[i])
        unique_transitions = len(set(transitions))
        max_possible = min(4, len(transitions))
        return 1.0 - (unique_transitions / max_possible)
        
    def get_weighted_random_strategy(self) -> str:
        strategies = []
        weights = []
        for strategy in self.available_strategies:
            success_rate = self.strategy_performance[strategy]["wins"] / max(1, self.strategy_performance[strategy]["uses"])
            weight = 0.1 + success_rate
            strategies.append(strategy)
            weights.append(weight)
        return random.choices(strategies, weights=weights)[0]
        
    def record_strategy_performance(self, strategy: str, won_round: bool):
        self.strategy_performance[strategy]["uses"] += 1
        if won_round:
            self.strategy_performance[strategy]["wins"] += 1


class PowerfulAdaptiveAI:
    """Powerful adaptive AI with PROPER adversarial search"""
    
    def __init__(self):
        self.name = "Adaptive AI"
        self.analyzer = AdvancedStrategyAnalyzer()
        self.adversarial_search = AdversarialSearchAI()
        self.fuzzy_system = FuzzyLogicSystem()
        self.my_history = []
        self.opp_history = []
        self.current_strategy = "minimax"
        self.strategy_change_cooldown = 0
        self.consecutive_losses = 0
        self.aggression_level = 0.3
        self.bayesian_prior = 0.5
        
    def update_bayesian_prior(self, move: str):
        if move == 'C':
            self.bayesian_prior = min(0.95, self.bayesian_prior * 1.1)
        else:
            self.bayesian_prior = max(0.05, self.bayesian_prior * 0.9)
            
    def bayesian_decision(self) -> str:
        if not self.opp_history:
            return 'C'
            
        recent_coop = self.opp_history[-3:].count('C') / min(3, len(self.opp_history))
        evidence_strength = min(1.0, len(self.opp_history) / 10)
        
        likelihood = recent_coop
        posterior = (likelihood * self.bayesian_prior) / (
            likelihood * self.bayesian_prior + (1 - likelihood) * (1 - self.bayesian_prior))
        
        confidence = abs(posterior - 0.5) * 2
        if confidence > 0.6:
            return 'C' if posterior > 0.5 else 'D'
        else:
            return self.minimax_decision()
            
    def pattern_matcher_decision(self) -> str:
        if len(self.opp_history) < 4:
            return self.minimax_decision()
            
        recent_pattern = ''.join(self.opp_history[-4:])
        pattern_responses = {
            "CCCC": 'D',
            "DDDD": 'C',  
            "CDCD": self.opp_history[-1],
            "DCDC": self.opp_history[-1],
            "CCDC": 'D',
            "DDCD": 'C',
        }
        
        if recent_pattern in pattern_responses:
            return pattern_responses[recent_pattern]
            
        return self.markov_prediction()
        
    def markov_prediction(self) -> str:
        if len(self.opp_history) < 3:
            return self.minimax_decision()
            
        current_state = ''.join(self.opp_history[-2:])
        transitions = {'C': 0, 'D': 0}
        
        for i in range(2, len(self.opp_history)):
            if ''.join(self.opp_history[i-2:i]) == current_state:
                next_move = self.opp_history[i]
                transitions[next_move] += 1
                
        total = sum(transitions.values())
        if total == 0:
            return self.minimax_decision()
            
        predicted_move = max(transitions, key=transitions.get)
        return 'D' if predicted_move == 'C' else 'C'
        
    def minimax_decision(self) -> str:
        """Use PROPER minimax with recursive min/max functions"""
        if len(self.my_history) < 2:
            if not self.opp_history:
                return 'C'
            coop_rate = self.opp_history.count('C') / len(self.opp_history)
            return 'C' if coop_rate > 0.6 else 'D'
            
        current_score = self.calculate_current_score()
        current_opp_score = self.calculate_opponent_score()
        
        # Use the PROPER minimax search
        best_move = self.adversarial_search.get_best_move(
            self.my_history, 
            self.opp_history,
            current_score,
            current_opp_score
        )
        
        return best_move if best_move else 'C'
        
    def fuzzy_decision(self) -> str:
        if not self.opp_history:
            return 'C'
            
        coop_rate = self.opp_history.count('C') / len(self.opp_history)
        pattern_consistency = self.analyzer.calculate_pattern_consistency(self.opp_history)
        
        my_total = self.calculate_current_score()
        opp_total = self.calculate_opponent_score()
        score_diff = my_total - opp_total
        
        cooperation_tendency = self.fuzzy_system.fuzzy_inference(coop_rate, pattern_consistency, score_diff)
        
        threshold = 0.5 + (cooperation_tendency - 0.5) * 0.3
        return 'C' if random.random() < threshold else 'D'
        
    def tit_for_tat_decision(self) -> str:
        if not self.opp_history:
            return 'C'
            
        if self.opp_history[-1] == 'D':
            overall_coop = self.opp_history.count('C') / len(self.opp_history)
            forgiveness_prob = 0.1 + (overall_coop * 0.3)
            if random.random() < forgiveness_prob:
                return 'C'
                
        return self.opp_history[-1]
        
    def adaptive_decision(self) -> str:
        if not self.opp_history:
            return 'C'
            
        success_rate = self.calculate_success_rate()
        
        if success_rate > 0.7:
            if self.my_history:
                return self.my_history[-1] if random.random() < 0.8 else 'D'
            else:
                return 'C'
        elif success_rate < 0.4:
            self.consecutive_losses += 1
            self.aggression_level = min(0.8, self.aggression_level + 0.1)
            if self.consecutive_losses > 2:
                return 'D'
            else:
                return 'C' if random.random() < 0.3 else 'D'
        else:
            self.consecutive_losses = 0
            return 'C' if random.random() < 0.6 else 'D'
            
    def calculate_success_rate(self) -> float:
        if len(self.my_history) < 2:
            return 0.5
            
        total_score = 0
        max_possible = 0
        
        for i in range(len(self.my_history)):
            my_move = self.my_history[i]
            opp_move = self.opp_history[i]
            payoff = self.adversarial_search.payoff_matrix[(my_move, opp_move)][0]
            total_score += payoff
            max_possible += 5
            
        base_rate = total_score / max_possible
        
        exploit_success = sum(1 for i in range(len(self.my_history)) 
                            if self.my_history[i] == 'D' and self.opp_history[i] == 'C') / max(1, len(self.my_history))
        
        comprehensive_success = (base_rate * 0.7) + (exploit_success * 0.3)
        return min(1.0, comprehensive_success)
    
    def calculate_current_score(self) -> int:
        total = 0
        for i in range(len(self.my_history)):
            my_move = self.my_history[i]
            opp_move = self.opp_history[i] if i < len(self.opp_history) else 'C'
            total += self.adversarial_search.payoff_matrix[(my_move, opp_move)][0]
        return total
    
    def calculate_opponent_score(self) -> int:
        total = 0
        for i in range(len(self.opp_history)):
            my_move = self.my_history[i] if i < len(self.my_history) else 'C'
            opp_move = self.opp_history[i]
            total += self.adversarial_search.payoff_matrix[(my_move, opp_move)][1]
        return total
        
    def decide_move(self, round_num: int, my_score: int, opp_score: int) -> str:
        should_analyze = (self.strategy_change_cooldown <= 0 and 
                         len(self.opp_history) >= 3 and 
                         (round_num % 3 == 0 or len(self.opp_history) < 8))
                         
        if should_analyze:
            new_strategy = self.analyzer.analyze_opponent(self.opp_history, my_score, opp_score)
            if new_strategy != self.current_strategy:
                self.current_strategy = new_strategy
                self.strategy_change_cooldown = random.randint(2, 5)
        else:
            self.strategy_change_cooldown = max(0, self.strategy_change_cooldown - 1)
            
        if random.random() < 0.02:
            return random.choice(['C', 'D'])
            
        if self.current_strategy == "minimax":
            move = self.minimax_decision()
        elif self.current_strategy == "fuzzy":
            move = self.fuzzy_decision()
        elif self.current_strategy == "tit_for_tat":
            move = self.tit_for_tat_decision()
        elif self.current_strategy == "adaptive":
            move = self.adaptive_decision()
        elif self.current_strategy == "pattern_matcher":
            move = self.pattern_matcher_decision()
        elif self.current_strategy == "bayesian":
            move = self.bayesian_decision()
        else:
            move = self.minimax_decision()
            
        self.update_bayesian_prior(move)
        return move
        
    def reset(self):
        self.my_history = []
        self.opp_history = []
        self.current_strategy = "minimax"
        self.strategy_change_cooldown = 0
        self.consecutive_losses = 0
        self.aggression_level = 0.3
        self.bayesian_prior = 0.5
        self.analyzer = AdvancedStrategyAnalyzer()
        self.adversarial_search = AdversarialSearchAI()
        self.fuzzy_system = FuzzyLogicSystem()


# Keep the StrategicOpponentAI class as it was
class StrategicOpponentAI:
    """Strategic opponent AI"""
    
    def __init__(self, ai_type: str):
        self.ai_type = ai_type
        self.name = f"{ai_type.replace('_', ' ').title()} AI"
        self.my_history = []
        self.opp_history = []
        self.personality = self.initialize_personality(ai_type)
        
    def initialize_personality(self, ai_type):
        personalities = {
            "cooperative": {"coop_bias": 0.8, "forgiveness": 0.9, "aggression": 0.1},
            "aggressive": {"coop_bias": 0.2, "forgiveness": 0.1, "aggression": 0.9},
            "random": {"coop_bias": 0.5, "forgiveness": 0.5, "aggression": 0.5},
            "tit_for_tat": {"coop_bias": 0.5, "forgiveness": 0.3, "aggression": 0.5},
            "forgiving": {"coop_bias": 0.7, "forgiveness": 0.8, "aggression": 0.2},
            "strategic": {"coop_bias": 0.6, "forgiveness": 0.4, "aggression": 0.6},
            "unpredictable": {"coop_bias": 0.5, "forgiveness": 0.5, "aggression": 0.5},
            "exploitative": {"coop_bias": 0.4, "forgiveness": 0.2, "aggression": 0.8},
            "mirror": {"coop_bias": 0.5, "forgiveness": 0.5, "aggression": 0.5},
        }
        return personalities.get(ai_type, personalities["random"])
    
    def get_strength(self) -> float:
        strengths = {
            "cooperative": 0.6, "aggressive": 0.8, "random": 0.4,
            "tit_for_tat": 0.7, "forgiving": 0.5, "strategic": 0.9,
            "unpredictable": 0.75, "exploitative": 0.85, "mirror": 0.7,
        }
        return strengths.get(self.ai_type, 0.5)
    
    def decide_move(self, round_num: int, my_score: int, opp_score: int) -> str:
        if self.ai_type == "cooperative":
            return 'C' if random.random() < 0.8 else 'D'
        elif self.ai_type == "aggressive":
            return 'D' if random.random() < 0.8 else 'C'
        elif self.ai_type == "random":
            return 'C' if random.random() < 0.5 else 'D'
        elif self.ai_type == "tit_for_tat":
            if not self.opp_history:
                return 'C'
            return self.opp_history[-1]
        elif self.ai_type == "forgiving":
            if not self.opp_history:
                return 'C'
            if self.opp_history[-1] == 'D' and random.random() < self.personality["forgiveness"]:
                return 'C'
            return self.opp_history[-1]
        elif self.ai_type == "strategic":
            if len(self.opp_history) < 3:
                return 'C'
            coop_rate = self.opp_history.count('C') / len(self.opp_history)
            return 'C' if coop_rate > 0.6 else 'D'
        elif self.ai_type == "unpredictable":
            if len(self.my_history) < 2:
                return 'C'
            if random.random() < 0.3:
                return 'D' if self.my_history[-1] == 'C' else 'C'
            else:
                return self.opp_history[-1] if self.opp_history else 'C'
        elif self.ai_type == "exploitative":
            if len(self.opp_history) < 4:
                return 'C'
            if all(move == 'C' for move in self.opp_history[-2:]):
                return 'D'
            return 'C' if random.random() < 0.4 else 'D'
        elif self.ai_type == "mirror":
            if not self.opp_history:
                return 'C'
            return self.opp_history[-1]
        else:
            return 'C'
            
    def reset(self):
        self.my_history = []
        self.opp_history = []
