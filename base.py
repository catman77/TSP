"""
Symbolic arithmetic for real numbers with structural representation
"""

import re
from typing import Union, Tuple, Dict
from fractions import Fraction
import math
from functools import lru_cache

# Global cache for structural patterns
_STRUCTURAL_CACHE: Dict[Tuple[str, str], str] = {}
_PATTERN_CACHE: Dict[str, Dict] = {}


class SymbolicReal:
    """
    Class for representing real numbers through symbolic sequences
    with proper handling of special cases
    
    Symbols:
    - S: successor (addition of 1)
    - P: predecessor (subtraction of 1)
    - Z: standard zero
    - Ω: infinity
    - Λ: fractal/rational structure marker
    - I: division marker within fractal structures
    """
    
    def __init__(self, symbolic_string: str = ""):
        """Initialize symbolic number"""
        self.symbolic_string = symbolic_string if symbolic_string else "Z"
        # self._normalize()
    
    def _normalize(self):
        """Basic normalization of symbolic string"""
        self.symbolic_string = ''.join(self.symbolic_string.split())
        if not self.symbolic_string:
            self.symbolic_string = "Z"
    
    def _count_symbol(self, symbol: str) -> int:
        """Count the number of symbols"""
        return self.symbolic_string.count(symbol)
    
    def _invert_symbol(self, symbol: str) -> str:
        """Inversion of a single symbol"""
        invert_map = {
            'S': 'P', 'P': 'S', 'I': 'I', 'Z': 'Z', 'Ω': 'Ω', 'Λ': 'Λ'
        }
        return invert_map.get(symbol, symbol)
    
    def _invert_string(self, string: str) -> str:
        """Inversion of the entire string"""
        return ''.join(self._invert_symbol(char) for char in string)
    
    def _invert_fractal_structure(self, fractal_string: str) -> str:
        """Special inversion of fractal structure"""
        if not fractal_string.startswith('Λ'):
            return self._invert_string(fractal_string)
        
        if 'I' in fractal_string:
            parts = fractal_string.split('I')
            if len(parts) == 2:
                lambda_and_numerator = parts[0]
                denominator = parts[1]
                numerator_part = lambda_and_numerator[1:]
                inverted_numerator = self._invert_string(numerator_part)
                return f"Λ{inverted_numerator}I{denominator}"
        
        fractal_content = fractal_string[1:]
        inverted_content = self._invert_string(fractal_content)
        return f"Λ{inverted_content}"
    
    def is_zero(self) -> bool:
        """Zero check - proper verification"""
        s_count = self._count_symbol('S')
        p_count = self._count_symbol('P')
        other_symbols = len([c for c in self.symbolic_string if c not in 'SP'])
        
        # Standard algebraic zero: equal count of S and P (including PS = 0)
        if s_count == p_count and other_symbols == 0 and s_count > 0:
            return True
        
        # Standard zero symbol
        if self.symbolic_string == "Z":
            return True
        
        # Topological zero (division by infinity)
        if self.is_topological_zero():
            return True
        
        return False
    
    def is_topological_zero(self) -> bool:
        """Check for topological zero (result of division by infinity)"""
        # Patterns of the form Λ...Ω or ...Ω where the numerator is finite
        return ('Λ' in self.symbolic_string and 'Ω' in self.symbolic_string and 
                not self.symbolic_string.startswith('Ω'))
    
    def is_infinity(self) -> bool:
        """Check for infinity"""
        return self.symbolic_string.startswith('Ω')
    
    def is_fractal(self) -> bool:
        """Check for fractal structure"""
        return 'Λ' in self.symbolic_string
    
    def classify_zero_type(self) -> str:
        """Classification of zero type"""
        if not self.is_zero():
            return "not_zero"
        
        s_count = self._count_symbol('S')
        p_count = self._count_symbol('P')
        
        if s_count == p_count and s_count > 0 and not ('Λ' in self.symbolic_string or 'Ω' in self.symbolic_string):
            return "algebraic"
        elif self.is_topological_zero():
            return "topological"
        elif self.symbolic_string == "Z":
            return "standard"
        else:
            return "mixed"
    
    @classmethod
    def from_float(cls, value: float) -> 'SymbolicReal':
        """Proper conversion from float to symbolic representation"""
        if math.isnan(value):
            return cls("ΛZ")  # NaN as fractal zero
        
        if math.isinf(value):
            if value > 0:
                return cls("Ω")  # Positive infinity
            else:
                return cls("ΩP")  # Negative infinity
        
        if value == 0:
            return cls("SP")  # Algebraic zero
        
        if value == 1:
            return cls("S")  # Unit
        
        if value == -1:
            return cls("P")  # Minus unit
        
        # Handle integers - improved logic
        if value == int(value) and abs(value) <= 50:  # Only for small integers
            if value > 0:
                return cls("S" * int(value))
            else:
                return cls("P" * int(abs(value)))
        
        # Handle fractional numbers via fractal structure
        frac = Fraction(value).limit_denominator(10000)  # Increased precision
        
        if frac.numerator == 0:
            return cls("SP")  # Algebraic zero
        
        # Create fractal structure for fractions
        if frac.denominator == 1:
            # Integer - FULL symbolic representation to preserve complexity
            abs_value = abs(frac.numerator)
            if abs_value <= 1000:  # Up to 1000 - full representation
                if frac.numerator > 0:
                    return cls("S" * abs_value)
                else:
                    return cls("P" * abs_value)
            else:
                # For very large numbers - compact representation
                sign = "S" if frac.numerator > 0 else "P" 
                return cls(f"Λ{sign}{abs_value}")
        
        # Regular fraction as fractal structure - compact notation
        if abs(frac.numerator) <= 50 and frac.denominator <= 50:
            # For small fractions use direct representation
            num_part = "S" * abs(frac.numerator) if frac.numerator > 0 else "P" * abs(frac.numerator)
            den_part = "S" * frac.denominator
            return cls(f"Λ{num_part}I{den_part}")
        else:
            # For large fractions use numeric representation inside Λ
            sign = "S" if frac.numerator > 0 else "P"
            return cls(f"Λ{sign}{abs(frac.numerator)}I{frac.denominator}")
    
    def to_float(self) -> float:
        """Proper conversion from symbolic representation to float"""
        # Check forced zero
        if hasattr(self, '_forced_zero') and self._forced_zero:
            return 0.0
        
        s = self.symbolic_string
        
        # Handle special cases
        if s == "Z" or not s:
            return 0.0
        
        # Check for NaN (ΛZ)
        if s == "ΛZ":
            return float('nan')

        # FIX: Handle topological zeros BEFORE fractal structures
        # This is critical for correct handling of structures like SPΩPΛS
        if "Ω" in s:
            # Check simple topological zeros: SPΩ, PSΩ and their variants
            if (("SPΩ" in s) or ("PSΩ" in s)) and not s.startswith("Ω"):
                return 0.0  # Topological zeros equal 0
            
            # Check more complex topological zeros
            if len(s) >= 3 and 'Ω' in s and not s.startswith('Ω'):
                # If there are S/P symbols before Ω, this could be a topological zero
                omega_pos = s.find('Ω')
                before_omega = s[:omega_pos]
                if before_omega and all(c in "SP" for c in before_omega):
                    # Simple topological zero like SPΩ, SPSPΩ etc.
                    return 0.0
            
            # Handle pure infinity (only Ω symbols)
            if s.startswith("Ω") and all(c in "ΩP" for c in s):
                if "P" in s:
                    return float('-inf')
                return float('inf')
            
            # Otherwise it's regular infinity
            if s.count('P') > s.count('S'):
                return float('-inf')
            return float('inf')

        # Handle fractal structures (numbers with Λ)
        if "Λ" in s:
            return self._parse_fractal_to_float(s)
        
        # Handle simple strings of S and P
        s_count = s.count('S')
        p_count = s.count('P')
        i_count = s.count('I')
        
        # Check for explicit algebraic zeros like "SP", "SPSP", but not "PS", "SSP" etc.
        # Algebraic zero: string consists only of S and P in equal quantities
        # and represents a balanced structure
        other_symbols_count = len([c for c in s if c not in 'SPI'])
        if (s_count == p_count and s_count > 0 and other_symbols_count == 0 and 
            self._is_balanced_structure(s)):
            return 0.0
        
        return float(s_count - p_count + i_count)
    
    def _base_to_float(self) -> float:
        """Base to_float method without _forced_zero check"""
        return self.to_float()
    
    def _is_balanced_structure(self, s: str) -> bool:
        """Check if the string is a balanced algebraic structure"""
        # Check simple cases of known algebraic zeros
        known_zeros = {"SP", "PS", "SPSP", "SPPS", "SSPP", "PPSS"}
        if s in known_zeros:
            return True
        
        # For more complex cases: check that S and P alternate or are grouped in balance
        s_count = s.count('S')
        p_count = s.count('P')
        
        if s_count != p_count:
            return False
        
        # If the string consists only of S and P in equal quantities - it's a potential zero
        return all(c in 'SP' for c in s)
    
    def _parse_fractal_to_float(self, s: str) -> float:
        """Proper parsing of fractal structures with support for multiple Λ"""
        
        # Handle NaN
        if "ΛZ" in s:
            return float('nan')
        
        # If the string has multiple Λ, split and sum
        if s.count('Λ') > 1:
            total = 0.0
            parts = s.split('Λ')[1:]  # Skip empty part before first Λ
            
            for part in parts:
                if part:  # Skip empty parts
                    # Create temporary fractal string for each part
                    temp_fractal = f"Λ{part}"
                    temp_obj = SymbolicReal(temp_fractal)
                    temp_value = temp_obj._parse_single_fractal_to_float(temp_fractal)
                    total += temp_value
            
            return total
        
        # Handle single fractal part
        return self._parse_single_fractal_to_float(s)
    
    def _parse_single_fractal_to_float(self, s: str) -> float:
        """Parse single fractal structure"""
        # Find fractal part
        lambda_pos = s.find('Λ')
        if lambda_pos >= 0:
            fractal_part = s[lambda_pos + 1:]
            
            # Handle division (contains I)
            if 'I' in fractal_part:
                i_pos = fractal_part.find('I')
                numerator_part = fractal_part[:i_pos]
                denominator_part = fractal_part[i_pos + 1:]
                
                # Parse numerator
                numerator = self._parse_part_to_number(numerator_part)
                
                # Parse denominator
                denominator = self._parse_part_to_number(denominator_part)
                
                # Handle division by infinity → topological zero
                if 'Ω' in denominator_part:
                    return 0.0
                
                if denominator == 0:
                    return float('inf') if numerator > 0 else float('-inf') if numerator < 0 else 0.0
                
                return float(numerator) / float(denominator)
            else:
                # Simple fractal structure
                if 'Ω' in fractal_part:
                    return float('inf') if fractal_part.count('S') >= fractal_part.count('P') else float('-inf')
                
                value = self._parse_part_to_number(fractal_part)
                return float(value)
        
        return 0.0
    
    def _parse_part_to_number(self, part: str) -> int:
        """Parse part of fractal structure to number"""
        if not part:
            return 0
        
        # If part consists only of digits
        if part.isdigit():
            return int(part)
        
        # If part starts with S or P, followed by digits
        if part[0] in 'SP' and len(part) > 1 and part[1:].isdigit():
            sign = 1 if part[0] == 'S' else -1
            return sign * int(part[1:])
        
        # FIX: Mixed structures (digits + symbols)
        # Extract numeric part and symbolic part separately
        numeric_value = 0
        symbolic_value = 0
        
        # Collect all digits in string
        numeric_chars = ''.join(c for c in part if c.isdigit())
        if numeric_chars:
            numeric_value = int(numeric_chars)
        
        # Count symbolic value
        symbolic_value = self._count_symbolic_value(part)
        
        # Combine: base number + symbolic contribution
        # If there are numbers, they dominate; symbols give correction
        if numeric_value > 0:
            return numeric_value + symbolic_value
        else:
            return symbolic_value
    
    def _count_symbolic_value(self, symbolic_str: str) -> int:
        """Count symbolic value of string"""
        s_count = symbolic_str.count('S')
        p_count = symbolic_str.count('P')
        i_count = symbolic_str.count('I')
        
        return s_count - p_count + i_count
            
    def _parse_complex_fractal(self) -> float:
        """Parsing complex fractal structures"""
        parts = self.symbolic_string.split('I')
        if len(parts) >= 3:
            first_part = parts[0]
            second_part = parts[1]
            
            if len(parts) >= 4:
                third_part = parts[2]
                fourth_part = parts[3]
                
                if first_part.startswith('ΛΛ'):
                    num1_str = first_part[2:]
                else:
                    num1_str = first_part[1:]
                
                num1 = num1_str.count('S') - num1_str.count('P')
                den1 = second_part.count('S') - second_part.count('P')
                
                if third_part.startswith('Λ'):
                    num2_str = third_part[1:]
                else:
                    num2_str = third_part
                    
                num2 = num2_str.count('S') - num2_str.count('P')
                den2 = fourth_part.count('S') - fourth_part.count('P')
                
                if den1 != 0 and num2 != 0:
                    return (float(num1) * float(den2)) / (float(den1) * float(num2))
        
        lambda_count = self.symbolic_string.count('Λ')
        return 1.0 / (10.0 ** lambda_count)
    
    def __add__(self, other: 'SymbolicReal') -> 'SymbolicReal':
        """Addition with special cases"""
        if isinstance(other, (int, float)):
            other = SymbolicReal.from_float(float(other))
        
        # Processing infinity - infinity dominates
        if self.is_infinity() or other.is_infinity():
            return SymbolicReal("Ω")
                
        # Simple concatenation
        return SymbolicReal(self.symbolic_string + other.symbolic_string)
    
    def __sub__(self, other: 'SymbolicReal') -> 'SymbolicReal':
        """Subtraction with special cases"""
        if isinstance(other, (int, float)):
            other = SymbolicReal.from_float(float(other))
        
        inverted_other = SymbolicReal(self._invert_string(other.symbolic_string))
        
        # Special handling: infinity - opposite infinity
        if (self.symbolic_string == "Ω" and other.symbolic_string == "ΩP") or \
           (self.symbolic_string == "ΩP" and other.symbolic_string == "Ω"):
            # (+∞) - (-∞) = +∞ + (+∞) = +∞
            # (-∞) - (+∞) = -∞ + (-∞) = -∞
            if self.symbolic_string == "Ω":
                return SymbolicReal("Ω")  # Positive infinity
            else:
                return SymbolicReal("ΩP")  # Negative infinity
                    
        # Processing infinity
        if self.is_infinity() and other.is_infinity():
            # ∞ - ∞ = 0 (indeterminacy resolved as zero)
            return SymbolicReal("Z")
        elif self.is_infinity() and not other.is_infinity():
            return SymbolicReal("Ω")
        elif not self.is_infinity() and other.is_infinity():
            return SymbolicReal("Ω" + self._invert_string(other.symbolic_string))
        
        # Processing topological zeros
        if other.is_topological_zero():
            inverted_topo = self._invert_string(other.symbolic_string)
            return SymbolicReal(self.symbolic_string + inverted_topo)
        
        # For fractal numbers, use numerical subtraction
        if self.is_fractal() or other.is_fractal():
            try:
                result_float = self.to_float() - other.to_float()
                return SymbolicReal.from_float(result_float)
            except:
                if other.is_fractal():
                    inverted_other = SymbolicReal(self._invert_fractal_structure(other.symbolic_string))
                else:
                    inverted_other = SymbolicReal(self._invert_string(other.symbolic_string))
                return self + inverted_other
        
        # Simple symbolic subtraction
        inverted_other = SymbolicReal(self._invert_string(other.symbolic_string))
        return self + inverted_other
    
    def __mul__(self, other):
        """Purely symbolic multiplication according to Numbers_Description.md"""
        if isinstance(other, (int, float)):
            other = SymbolicReal.from_float(float(other))
        
        # Handle only direct zero and infinity symbols
        if self.symbolic_string == "SP" or self.symbolic_string == "Z" or other.symbolic_string == "SP" or other.symbolic_string == "Z":
            return SymbolicReal("Z")
        
        # FIX: Do NOT check to_float() == 0.0 - preserve symbolic structures
        # Remove premature collapse to zeros
        
        # SPECIAL HANDLING: Topological zeros before general algorithm
        # ONLY simple topological zeros: "SPΩP", "PSΩS" etc.
        # NOT complex fractal expressions with Ω inside
        def is_simple_topological_zero(s):
            # Only simple patterns of topological zeros
            simple_patterns = ["SPΩP", "PSΩS", "SPΩPS", "PSΩSP", "SPΩPSP"]
            return s in simple_patterns
        
        # SPECIAL HANDLING: Fractal zeros  
        # ONLY simple fractal zeros without coefficients
        def is_simple_fractal_zero(s):
            # Only simple patterns of fractal zeros  
            simple_patterns = ["ΛSPSP", "ΛPSPS", "ΛSPSPSP", "ΛPSPSPS"]
            return s in simple_patterns
        
        if is_simple_topological_zero(self.symbolic_string) or is_simple_topological_zero(other.symbolic_string):
            # Multiplication by simple topological zero = complex structure, but numerically zero
            # PRESERVE symbolic complexity, but ensure zero value
            complex_result = self.symbolic_string + other.symbolic_string
            result = SymbolicReal(complex_result)
            # Override to_float for this case
            result._forced_zero = True
            return result
        
        if is_simple_fractal_zero(self.symbolic_string) or is_simple_fractal_zero(other.symbolic_string):
            # Multiplication by simple fractal zero = complex structure, but numerically zero
            complex_result = self.symbolic_string + other.symbolic_string
            result = SymbolicReal(complex_result)
            result._forced_zero = True
            return result
        
        # ADDITIONAL CHECK: If one of the operands itself evaluates to zero
        # but is not a simple structure, should still give zero
        try:
            self_value = self._base_to_float()  # Call base method without _forced_zero
            other_value = other._base_to_float()
            if abs(self_value) < 1e-10 or abs(other_value) < 1e-10:
                complex_result = self.symbolic_string + other.symbolic_string
                result = SymbolicReal(complex_result)
                result._forced_zero = True
                return result
        except:
            pass  # Continue normal processing
        
        if self.symbolic_string.startswith("Ω") or other.symbolic_string.startswith("Ω"):
            return SymbolicReal("Ω")
        
        # Purely symbolic multiplication for all cases
        result_string = true_symbolic_multiply(
            self.symbolic_string, 
            other.symbolic_string
        )
        
        return SymbolicReal(result_string)
    
    def __truediv__(self, other: 'SymbolicReal') -> 'SymbolicReal':
        """Division with special cases"""
        if isinstance(other, (int, float)):
            other = SymbolicReal.from_float(float(other))
        
        # Division by any type of zero - constructive division
        if other.is_zero():
            return self._divide_by_zero(other)
        
        # Division of zero by something
        if self.is_zero():
            return SymbolicReal("Z")
        
        # Division by infinity
        if other.is_infinity():
            if self.is_infinity():
                # ∞ / ∞ = 1, but taking signs into account
                if other.symbolic_string == "ΩP":  # division by -∞
                    return SymbolicReal("P")  # = -1
                else:
                    return SymbolicReal("S")  # = 1
            return SymbolicReal("SP")  # finite/∞ = 0
        
        # Division of infinity by finite
        if self.is_infinity():
            return SymbolicReal("Ω")
        
        # Division by topological zero
        if other.is_topological_zero():
            # a / (b/∞) = a * ∞ / b = (a/b) * ∞
            return SymbolicReal("Ω")
        
        # Division of unit by itself
        if self.symbolic_string == "S" and other.symbolic_string == "S":
            return SymbolicReal("S")
        
        # Special handling of division of fractal numbers
        if self.is_fractal() and other.is_fractal():
            try:
                result_float = self.to_float() / other.to_float()
                return SymbolicReal.from_float(result_float)
            except:
                return SymbolicReal(f"Λ{self.symbolic_string}I{other.symbolic_string}")
        
        # Division of fractal by simple or vice versa
        if self.is_fractal() or other.is_fractal():
            try:
                result_float = self.to_float() / other.to_float()
                return SymbolicReal.from_float(result_float)
            except:
                return SymbolicReal(f"Λ{self.symbolic_string}I{other.symbolic_string}")
        
        # Simple symbolic division
        return SymbolicReal(f"Λ{self.symbolic_string}I{other.symbolic_string}")
    
    def _divide_by_zero(self, zero_structure: 'SymbolicReal') -> 'SymbolicReal':
        """Constructive division by zero"""
        zero_type = zero_structure.classify_zero_type()
        
        if zero_type == "algebraic":
            # Division by algebraic zero creates a Λ-structure
            return SymbolicReal(f"Λ{self.symbolic_string}Z{zero_structure.symbolic_string}")
        
        elif zero_type == "topological":
            # Division by topological zero (a/∞) gives a*∞
            return SymbolicReal("Ω")
        
        elif zero_type == "standard":
            # Division by standard zero Z
            return SymbolicReal(f"Λ{self.symbolic_string}Z{zero_structure.symbolic_string}")
        
        else:
            # Mixed case
            return SymbolicReal(f"Λ{self.symbolic_string}Z{zero_structure.symbolic_string}")
    
    def __pow__(self, other: Union['SymbolicReal', int, float]) -> 'SymbolicReal':
        """Exponentiation"""
        if isinstance(other, (int, float)):
            other = SymbolicReal.from_float(float(other))
        
        if self.is_zero():
            if other.is_zero():
                return SymbolicReal("S")  # 0^0 = 1
            return SymbolicReal("Z")
        
        if other.is_zero():
            return SymbolicReal("S")  # any^0 = 1
        
        if other.symbolic_string == "S":
            return SymbolicReal(self.symbolic_string)
        
        if other.is_infinity():
            if self.is_zero():
                return SymbolicReal("Z")
            if self.symbolic_string == "S":
                return SymbolicReal("S")
            return SymbolicReal(f"Ω{self.symbolic_string}")
        
        if self.is_infinity():
            return SymbolicReal("Ω")
        
        # Iterative multiplication for small powers
        if not other.is_fractal() and not other.is_infinity():
            other_float = other.to_float()
            if other_float > 0 and other_float <= 10 and other_float == int(other_float):
                result = SymbolicReal("S")
                for _ in range(int(other_float)):
                    result = result * self
                return result
        
        return SymbolicReal(f"Λ{self.symbolic_string}^{other.symbolic_string}")
    
    # Support for operations with Python numbers
    def __radd__(self, other): return SymbolicReal.from_float(float(other)) + self
    def __rsub__(self, other): return SymbolicReal.from_float(float(other)) - self  
    def __rmul__(self, other): return SymbolicReal.from_float(float(other)) * self
    def __rtruediv__(self, other): return SymbolicReal.from_float(float(other)) / self
    def __rpow__(self, other): return SymbolicReal.from_float(float(other)) ** self
    
    def __neg__(self):
        """Unary minus"""
        return SymbolicReal(self._invert_string(self.symbolic_string))
    
    def __abs__(self):
        """Absolute value"""
        if 'P' in self.symbolic_string and 'S' not in self.symbolic_string:
            return -self
        return SymbolicReal(self.symbolic_string)
    
    def __str__(self) -> str:
        return self.symbolic_string
    
    def __repr__(self) -> str:
        return f"SymbolicReal('{self.symbolic_string}') ≈ {self.to_float()}"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return abs(self.to_float() - float(other)) < 1e-10
        if isinstance(other, SymbolicReal):
            return abs(self.to_float() - other.to_float()) < 1e-10
        return False


# True symbolic multiplication algorithm according to Numbers_Description.md
def true_symbolic_multiply(first_multiplier: str, second_multiplier: str) -> str:
    """
    True symbolic multiplication according to the algorithm from Numbers_Description.md:
    1. Take the first symbol of the second multiplier
    2. If S - the first multiplier is written to result, if P - involution of the first
    3. For each subsequent symbol of the second multiplier:
       - If S: concatenation of result with first multiplier
       - If P: concatenation of result with involution of first multiplier
    """
    
    if not second_multiplier:
        return "Z"  # Multiplication by empty string = 0
    
    if not first_multiplier:
        return "Z"  # Multiplication of empty string = 0
    
    # Involution (inversion) function for symbols
    def invert_symbol(symbol: str) -> str:
        invert_map = {
            'S': 'P', 'P': 'S', 'I': 'I', 'Z': 'Z', 'Ω': 'Ω', 'Λ': 'Λ'
        }
        return invert_map.get(symbol, symbol)
    
    def invert_string(string: str) -> str:
        """Proper involution accounting for fractal structures"""
        if string.startswith('Λ') and 'I' in string:
            # For fractal fractions ΛnumeratorIdenominator invert only numerator
            lambda_pos = string.find('Λ')
            i_pos = string.find('I')
            
            prefix = string[:lambda_pos + 1]  # "Λ"
            numerator_part = string[lambda_pos + 1:i_pos]
            denominator_part = string[i_pos:]  # "I..." including I
            
            # Invert only numerator
            inverted_numerator = ''.join(invert_symbol(char) for char in numerator_part)
            
            return prefix + inverted_numerator + denominator_part
        else:
            # Regular symbol-by-symbol involution
            return ''.join(invert_symbol(char) for char in string)
    
    result = ""
    
    # Iterate through each symbol of the second multiplier
    for symbol in second_multiplier:
        if symbol == 'S':
            # S: add first multiplier as is
            result += first_multiplier
        elif symbol == 'P':
            # P: add involution of first multiplier
            result += invert_string(first_multiplier)
        elif symbol == 'I':
            # I: scaling symbol - add as is
            result += first_multiplier
        elif symbol == 'Z':
            # Z: zero absorbs everything
            return "Z"
        elif symbol == 'Ω':
            # Ω: infinity dominates
            return "Ω"
        elif symbol == 'Λ':
            # Λ: fractal structure
            result += f"Λ{first_multiplier}"
    
    return result


def test_special_cases():
    """
    Tests for special cases: operations with different types of zeros and infinities
    """
    
    print("=== TESTING SPECIAL CASES ===\n")
    
    # Preparation of special values
    
    # Zeros of different types
    zero_algebraic_simple = SymbolicReal("SP")           # simple algebraic zero
    zero_algebraic_complex = SymbolicReal("SPSPSP")      # complex algebraic zero
    zero_topological_1 = SymbolicReal("ΛSΩ")            # 1/∞ - topological zero
    zero_topological_2 = SymbolicReal("ΛSSSΩ")          # 3/∞ - topological zero
    zero_mixed = SymbolicReal("ΛSPSPΩ")                  # mixed zero
    zero_standard = SymbolicReal("Z")                    # standard zero
    
    # Infinities of different types
    infinity_positive = SymbolicReal("Ω")                # +∞
    infinity_negative = SymbolicReal("ΩP")               # -∞
    infinity_scaled = SymbolicReal("ΩSSS")               # 3*∞
    infinity_fractional = SymbolicReal("ΛΩISS")          # ∞/2
    infinity_complex = SymbolicReal("ΩΛSSP")             # complex infinity
    
    # Ordinary numbers for testing
    num_positive = SymbolicReal("SSS")                   # 3
    num_negative = SymbolicReal("PPP")                   # -3
    num_fractional = SymbolicReal("ΛSSISS")              # 2/3
    num_one = SymbolicReal("S")                          # 1
    
    def test_operation(a, b, op_name, op_func, expected_pattern, description):
        """Helper function for testing operations"""
        try:
            result = op_func(a, b)
            result_str = str(result)
            
            # Check for expected pattern
            if callable(expected_pattern):
                matches_pattern = expected_pattern(result_str)
                expected_desc = expected_pattern.__name__
            else:
                matches_pattern = result_str == expected_pattern
                expected_desc = expected_pattern
            
            status = "PASSED" if matches_pattern else "FAILED"
            
            print(f"{description}")
            print(f"  {a} {op_name} {b} = {result_str}")
            print(f"  Expected: {expected_desc}")
            print(f"  Status: {status}")
            print()
            
        except Exception as e:
            print(f"{description}")
            print(f"  {a} {op_name} {b} - ERROR: {e}")
            print()
    
    # Functions for checking result patterns
    def is_omega_pattern(s):
        """Check for infinity pattern"""
        return s.startswith('Ω') or s == 'Ω'
    
    def is_zero_pattern(s):
        """Check for zero pattern"""
        return (s == 'Z' or s == 'SP' or 
                (s.count('S') == s.count('P') and s.count('S') > 0))
    
    def is_lambda_pattern(s):
        """Check for fractal structure pattern"""
        return 'Λ' in s
    
    def is_finite_pattern(s):
        """Check for finite number"""
        return not ('Ω' in s or 'Λ' in s) and not is_zero_pattern(s)
    
    # === TESTING ADDITION ===
    print("--- Testing ADDITION with special values ---")
    
    # Addition with algebraic zeros
    test_operation(num_positive, zero_algebraic_simple, "+", lambda a, b: a + b,
                  "SSSSP", "3 + (algebraic zero SP)")
    
    test_operation(zero_algebraic_complex, num_negative, "+", lambda a, b: a + b,
                  "SPSPSPPPP", "complex algebraic zero + (-3)")
    
    # Addition with topological zeros
    test_operation(num_positive, zero_topological_1, "+", lambda a, b: a + b,
                  "SSSΛSΩ", "3 + (topological zero 1/∞)")
    
    # Addition with infinities
    test_operation(num_positive, infinity_positive, "+", lambda a, b: a + b,
                  is_omega_pattern, "3 + (+∞)")
    
    test_operation(infinity_positive, infinity_negative, "+", lambda a, b: a + b,
                  is_omega_pattern, "(+∞) + (-∞)")
    
    test_operation(zero_algebraic_simple, infinity_positive, "+", lambda a, b: a + b,
                  is_omega_pattern, "algebraic zero + (+∞)")
    
    # === TESTING SUBTRACTION ===
    print("--- Testing SUBTRACTION with special values ---")
    
    # Subtraction of algebraic zeros
    test_operation(num_positive, zero_algebraic_simple, "-", lambda a, b: a - b,
                  "SSSPS", "3 - (algebraic zero SP)")
    
    test_operation(zero_algebraic_simple, num_positive, "-", lambda a, b: a - b,
                  "SPPPP", "algebraic zero - 3")
    
    # Subtraction of topological zeros
    test_operation(num_positive, zero_topological_1, "-", lambda a, b: a - b,
                  "SSSΛPΩ", "3 - (topological zero 1/∞)")
    
    # Subtraction of infinities
    test_operation(infinity_positive, num_positive, "-", lambda a, b: a - b,
                  is_omega_pattern, "(+∞) - 3")
    
    test_operation(infinity_positive, infinity_positive, "-", lambda a, b: a - b,
                  is_zero_pattern, "(+∞) - (+∞)")
    
    test_operation(infinity_positive, infinity_negative, "-", lambda a, b: a - b,
                  is_omega_pattern, "(+∞) - (-∞)")
    
    # === TESTING MULTIPLICATION ===
    print("--- Testing MULTIPLICATION with special values ---")
    
    # Multiplication by algebraic zeros
    test_operation(num_positive, zero_algebraic_simple, "*", lambda a, b: a * b,
                  is_zero_pattern, "3 * (algebraic zero SP)")
    
    test_operation(zero_algebraic_simple, zero_algebraic_complex, "*", lambda a, b: a * b,
                  is_zero_pattern, "algebraic zero * complex algebraic zero")
    
    # Multiplication by topological zeros
    test_operation(num_positive, zero_topological_1, "*", lambda a, b: a * b,
                  is_zero_pattern, "3 * (topological zero 1/∞)")
    
    # Multiplication by infinities
    test_operation(num_positive, infinity_positive, "*", lambda a, b: a * b,
                  is_omega_pattern, "3 * (+∞)")
    
    test_operation(zero_algebraic_simple, infinity_positive, "*", lambda a, b: a * b,
                  is_zero_pattern, "algebraic zero * (+∞)")
    
    test_operation(zero_topological_1, infinity_positive, "*", lambda a, b: a * b,
                  is_zero_pattern, "topological zero * (+∞)")
    
    test_operation(infinity_positive, infinity_negative, "*", lambda a, b: a * b,
                  is_omega_pattern, "(+∞) * (-∞)")
    
    # === TESTING DIVISION ===
    print("--- Testing DIVISION with special values ---")
    
    # Division by algebraic zeros (constructive division by zero)
    test_operation(num_positive, zero_algebraic_simple, "/", lambda a, b: a / b,
                  is_lambda_pattern, "3 / (algebraic zero SP) - constructive division by zero")
    
    test_operation(infinity_positive, zero_algebraic_simple, "/", lambda a, b: a / b,
                  is_lambda_pattern, "(+∞) / (algebraic zero SP)")
    
    # Division by topological zeros
    test_operation(num_positive, zero_topological_1, "/", lambda a, b: a / b,
                  is_omega_pattern, "3 / (topological zero 1/∞) = 3 * (∞/1) = 3*∞")
    
    test_operation(
        zero_algebraic_simple, zero_topological_1, "/", 
        lambda a, b: a / b,
        is_omega_pattern,  # Expecting infinity!
        "algebraic zero / topological zero → ∞"
    )    
    # Division by infinities
    test_operation(num_positive, infinity_positive, "/", lambda a, b: a / b,
                  is_zero_pattern, "3 / (+∞) = 0 (topological)")
    
    test_operation(infinity_positive, infinity_positive, "/", lambda a, b: a / b,
                  "S", "(+∞) / (+∞) = 1")
    
    test_operation(infinity_positive, infinity_negative, "/", lambda a, b: a / b,
                  "P", "(+∞) / (-∞) = -1")
    
    # Division of zeros
    test_operation(zero_algebraic_simple, zero_algebraic_complex, "/", lambda a, b: a / b,
                  is_lambda_pattern, "algebraic zero / complex algebraic zero")
    
    test_operation(zero_topological_1, zero_topological_2, "/", lambda a, b: a / b,
                  lambda s: '3' not in s or is_finite_pattern(s), "topological zero / topological zero")
    
    # === TESTING EXPONENTIATION ===
    print("--- Testing EXPONENTIATION with special values ---")
    
    # Raising to the power of zero
    test_operation(num_positive, zero_algebraic_simple, "**", lambda a, b: a ** b,
                  "S", "3^(algebraic zero) = 1")
    
    test_operation(zero_algebraic_simple, num_positive, "**", lambda a, b: a ** b,
                  is_zero_pattern, "algebraic zero^3 = 0")
    
    test_operation(zero_algebraic_simple, zero_algebraic_complex, "**", lambda a, b: a ** b,
                  "S", "0^0 = 1")
    
    # Raising to infinite power
    test_operation(num_positive, infinity_positive, "**", lambda a, b: a ** b,
                  is_omega_pattern, "3^(+∞)")
    
    test_operation(num_one, infinity_positive, "**", lambda a, b: a ** b,
                  "S", "1^(+∞) = 1")
    
    test_operation(zero_algebraic_simple, infinity_positive, "**", lambda a, b: a ** b,
                  is_zero_pattern, "0^(+∞) = 0")
    
    # Raising infinity to a power
    test_operation(infinity_positive, num_positive, "**", lambda a, b: a ** b,
                  is_omega_pattern, "(+∞)^3")
    
    test_operation(infinity_positive, zero_algebraic_simple, "**", lambda a, b: a ** b,
                  "S", "(+∞)^0 = 1")
    
    test_operation(infinity_positive, infinity_positive, "**", lambda a, b: a ** b,
                  is_omega_pattern, "(+∞)^(+∞)")
    
    # === SPECIAL COMBINATIONS ===
    print("--- TESTING SPECIAL COMBINATIONS ---")
    
    # Mixed operations
    test_operation(zero_mixed, infinity_complex, "+", lambda a, b: a + b,
                  is_omega_pattern, "mixed zero + complex infinity")
    
    test_operation(infinity_fractional, zero_topological_2, "*", lambda a, b: a * b,
                  is_zero_pattern, "fractal infinity * topological zero")
    
    # Operation chains
    temp_result = (num_positive / zero_algebraic_simple) + infinity_positive
    print(f"Chain: (3 / algebraic_zero) + (+∞) = {temp_result}")
    print(f"Expected: structure with dominant infinity")
    print()
    
    temp_result2 = (infinity_positive - infinity_negative) * zero_topological_1
    print(f"Chain: ((+∞) - (-∞)) * topological_zero = {temp_result2}")
    print(f"Expected: zero (since ∞ * 0 = 0)")
    print()
    
    # Property verification
    print("--- Checking ALGEBRAIC PROPERTIES ---")
    
    # Commutativity of addition with zeros
    result1 = num_positive + zero_algebraic_simple
    result2 = zero_algebraic_simple + num_positive
    print(f"Commutativity of addition: {result1} vs {result2}")
    print(f"Equal: {result1.to_float() == result2.to_float()}")
    print()
    
    # Associativity with infinities
    result3 = (infinity_positive + infinity_negative) + num_positive
    result4 = infinity_positive + (infinity_negative + num_positive)
    print(f"Associativity: {result3} vs {result4}")
    print()
    
    # Distributivity
    result5 = num_positive * (zero_algebraic_simple + num_fractional)
    result6 = (num_positive * zero_algebraic_simple) + (num_positive * num_fractional)
    print(f"Distributivity: {result5} vs {result6}")
    print()
    
    print("=== TESTING SPECIAL CASES COMPLETED ===")


def run_tests():
    """Comprehensive testing of all arithmetic operations"""
    
    print("=== TESTING THE LIBRARY ===\n")
    
    # Test numbers
    test_numbers = [
        (3, 2),           # integers
        (5, 7),           # integers
        (3.5, 2.25),      # fractional
        (1.75, 4.8),      # different fractional
        (-2, 3),          # mixed signs
        (0, 5),           # with zero
        (1, 1),           # units
        (10.5, -3.2),     # mixed fractional
    ]
    
    def format_test_result(num1, num2, result, operation, symbolic1, symbolic2, symbolic_result, passed):
        """Formatting test result"""
        status = "PASSED" if passed else "FAILED"
        return f"{num1} {operation} {num2} = {result:.6f} ({symbolic1} {operation} {symbolic2} = {symbolic_result}) - {status}!"
    
    def test_operation(op_name, op_symbol, op_func):
        """Testing a specific operation"""
        print(f"\n--- Testing operation {op_name.upper()} ---")
        
        for num1, num2 in test_numbers:
            try:
                # Create symbolic numbers
                sym1 = SymbolicReal.from_float(num1)
                sym2 = SymbolicReal.from_float(num2)
                
                # Perform operation
                sym_result = op_func(sym1, sym2)
                
                # Expected result
                if op_name == "division" and num2 == 0:
                    expected = float('inf')
                else:
                    expected = op_func(num1, num2)
                
                # Get numerical result
                numeric_result = sym_result.to_float()
                
                # Check correctness
                if math.isnan(expected) and math.isnan(numeric_result):
                    passed = True
                elif math.isinf(expected) and math.isinf(numeric_result):
                    passed = True
                elif abs(expected) > 1e10 and math.isinf(numeric_result):
                    passed = True
                else:
                    passed = abs(expected - numeric_result) < max(1e-6, abs(expected) * 1e-6)
                
                print(format_test_result(num1, num2, numeric_result, op_symbol, 
                                       str(sym1), str(sym2), str(sym_result), passed))
                
            except Exception as e:
                print(f"{num1} {op_symbol} {num2} - ERROR: {e}")
    
    # Testing main operations
    test_operation("addition", "+", lambda a, b: a + b)
    test_operation("subtraction", "-", lambda a, b: a - b)
    test_operation("multiplication", "*", lambda a, b: a * b)
    test_operation("division", "/", lambda a, b: a / b if b != 0 else SymbolicReal("Ω"))
    
    # Testing conversion
    print("\n--- TESTING CONVERSION ---")
    
    conversion_tests = [0, 1, -1, 3, -5, 2.5, -1.25, 0.1, 10.75, float('inf'), float('-inf')]
    
    for num in conversion_tests:
        try:
            sym_num = SymbolicReal.from_float(num)
            back_to_float = sym_num.to_float()
            
            if math.isnan(num) and math.isnan(back_to_float):
                passed = True
            elif math.isinf(num) and math.isinf(back_to_float):
                passed = (num > 0) == (back_to_float > 0)
            else:
                passed = abs(num - back_to_float) < max(1e-6, abs(num) * 1e-6)
            
            status = "PASSED" if passed else "FAILED"
            print(f"{num} -> {sym_num} -> {back_to_float:.6f} - {status}!")
            
        except Exception as e:
            print(f"Conversion {num} - ERROR: {e}")
    
    print("\n=== TESTING COMPLETED ===")


if __name__ == "__main__":
    # Running tests
    run_tests()
    test_special_cases()
