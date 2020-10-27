#include <assert.h>
#include <ctype.h>
#include <functional>
#include <gmp.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

/*
class ParseError(Exception):
  pass
#
# Default parsing functions give errors
#

def NullError(p, token, bp):
  raise ParseError("%s can't be used in prefix position" % token)

*/

/*
def LeftError(p, token, left, rbp):
# Hm is this not called because of binding power?
raise ParseError("%s can't be used in infix position" % token)
*/

struct Token {

  constexpr Token(const Token &other) : type{other.type} { *this = other; }
  constexpr Token &operator=(const Token &other) {
    type = other.type;
    switch (type) {
    case Type::Integer:
      value = other.value;
      break;
    case Type::String:
      string = other.string;
      break;
    default:
      break;
    }

    return *this;
  }
  ~Token() {
    switch (type) {
    case Type::String:
      string.~shared_ptr<char[]>();
      break;

    case Type::BigInt:
      mpz_clear(bigint);
      break;
    default:
      break;
    }
  }

  constexpr Token(long i) : value{i}, type{Type::Integer} {}
  union {
    long value;
    mpz_t bigint;
    std::shared_ptr<char[]> string;
  };
  enum class Type {
    Eof,
    Integer,
    String,
    Plus,
    Sub,
    Mul,
    Div,
    Increment,
    Decrement,
    AddAssign,
    MulAssign,
    SubAssign,
    DivAssign,
    ModAssign,
    AndAssign,
    Assign,
    OrAssign,
    Tilde,
    TildeAssign,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Mod,
    Ampersand,
    And,
    Pipe,
    Or,
    Not,
    LeftParen,
    RightParen,
    BigInt
  } type;

  Token(const std::shared_ptr<char[]> _string)
      : string{_string}, type{Type::String} {}
  constexpr Token(const Token::Type t) : value{0}, type{t} {}

  bool operator==(Token const &o) const {
    return type == o.type && value == o.value;
  }

  void print() const {
    switch (type) {
    case Type::Integer:
      printf("%li ", value);
      break;
    case Type::String:
      printf("%s ", string.get());
      break;
    case Type::Plus:
      printf("+ ");
      break;
    case Type::Mul:
      printf("* ");
      break;
    default:
      abort();
    }
  }
};

/*

#
# Input
#

class Token:
def __init__(self, type, val, loc=None):
self.type = type
self.val = val

def __repr__(self):
return '<Token %s %s>' % (self.type, self.val)


#
# Using the pattern here: http://effbot.org/zone/xml-scanner.htm
#

# NOTE: () and [] need to be on their own so (-1+2) works
TOKEN_RE = re.compile("""""", re.VERBOSE)

def Tokenize(s):
for item in TOKEN_RE.findall(s):
if item[0]:
  typ = 'number'
  val = int(item[0])
elif item[1]:
  typ = 'name'
  val = item[1]
elif item[2]:
  typ = item[2]
  val = item[2]
elif item[3]:
  typ = item[3]
  val = item[3]
yield Token(typ, val, loc=(0, 0))
*/

/*


#
# Simple and Composite AST nodes
#

class Node(object):
  def __init__(self, token):
    """
    Args:
      type: token type (operator, etc.)
      val: token val, only important for number and string
    """
    self.token = token

  def __repr__(self):
    return str(self.token.val)

*/

struct Node {
  explicit Node(struct Token t) : t{t} { type = Type::Leaf; }

  Node(const Node &n) { *this = n; }

  Node(struct Token t, const Node &n) {
    type = Type::Tree;
    N1 = NodeTree(t, n);
  }

  Node &operator=(const Node &n) {
    type = n.type;
    if (type == Type::Leaf) {
      t = n.t;
    } else {
      N1 = n.N1;
    }
    return *this;
  }

  ~Node() {
    switch (type) {
    case Type::Leaf:
      t.~Token();
      break;
    case Type::Tree:
      N1.~NodeTree();
      break;
    }
  }
  struct NodeTree {
    NodeTree &operator=(const NodeTree &n) {
      N1 = n.N1;
      length = n.length;
      return *this;
    }

    template <class... Args>
    NodeTree(Args &&...args)
        : N1{std::make_shared<struct Node[]>(std::forward<Args>(args)...)},
          length{sizeof...(Args)} {}

    std::shared_ptr<struct Node[]> N1; // array of operators, length long.
    size_t length;
    struct Node &operator[](size_t i) {
      if (i < length) {
        return N1[i];
      } else {
        abort();
      }
    }
    const struct Node &operator[](size_t i) const {
      if (i < length) {
        return N1[i];
      } else {
        abort();
      }
    }
  };
  enum class Type { Leaf, Tree } type;
  union {
    struct Token t;
    struct NodeTree N1;
  };
  void print() const {
    switch (type) {
      // Handle Literals here.
    case Node::Type::Leaf:
      t.print();
      break;
      // General Operators here.
    case Node::Type::Tree:
      printf("(");
      t.print();
      for (size_t i = 0; i < N1.length; i++) {
        N1[i].print();
      }
      printf(")");
    }
  }
};

/*
class CompositeNode(Node):
  def __init__(self, token, children):
    """
    Args:
      type: token type (operator, etc.)
    """
    Node.__init__(self, token)
    self.children = children

  def __repr__(self):
    args = ''.join([" " + repr(c) for c in self.children])
    return "(" + self.token.type + args + ")"
*/

/*
#
# Parser definition
#

class LeftInfo(object):
"""Row for operator.
In C++ this should be a big array.
"""
def __init__(self, led=None, lbp=0, rbp=0):
self.led = led or LeftError
self.lbp = lbp
self.rbp = rbp


class NullInfo(object):
"""Row for operator.
In C++ this should be a big array.
"""
def __init__(self, nud=None, bp=0):
self.nud = nud or NullError
self.bp = bp
*/

class BindingPower {
public:
  BindingPower(int _bp) : bp{_bp} {}
  int bp;
  operator int() const { return bp; }
};
class LeftBindingPower : public BindingPower {
public:
  LeftBindingPower(int _bp) : BindingPower{_bp} {}
  operator int() const { return bp; }
};
class RightBindingPower : public BindingPower {
public:
  RightBindingPower(int _bp) : BindingPower{_bp} {}
  operator int() const { return bp; }
};
class Parser;
using NullDenotation =
    std::function<Node(Parser &, struct Token, BindingPower)>;
using LeftDenotation =
    std::function<Node(Parser &, struct Token, Node &, RightBindingPower)>;

struct NullInfo {
  NullInfo() = default;
  NullInfo(const NullInfo &) = default;
  NullInfo(NullInfo &&) = default;
  NullInfo &operator=(const NullInfo &other) = default;
  NullInfo &operator=(NullInfo &&other) = default;
  ~NullInfo() = default;

  NullInfo(BindingPower _bp, NullDenotation _nud = NullError)
      : bp{_bp}, nud{_nud} {}
  BindingPower bp{-1};
  NullDenotation nud{NullError};

  static NullDenotation NullError;
};
NullDenotation NullInfo::NullError = [](Parser &, struct Token t,
                                        BindingPower) {
  t.print();
  printf(" cannot be used in prefix position!\n");
  abort();
  return Node{t};
};

struct LeftInfo {
  LeftInfo() = default;
  LeftInfo(LeftBindingPower _lbp, RightBindingPower _rbp,
           LeftDenotation _led = LeftError)
      : led{_led}, lbp{_lbp}, rbp{_rbp} {}
  LeftDenotation led{LeftError};
  LeftBindingPower lbp{-1};
  RightBindingPower rbp{-1};

  static LeftDenotation LeftError;
};
LeftDenotation LeftInfo::LeftError = [](Parser &, struct Token t, Node &,
                                        RightBindingPower) {
  t.print();
  printf(" cannot be used in infix position!\n");
  abort();
  return Node{t};
};

class ParserSpec {
public:
  void Null(BindingPower bp, NullDenotation nud, Token token) {
    null_lookup.emplace(token.type, NullInfo{bp, nud});
    left_lookup.try_emplace(token.type, LeftInfo(-1, -1));
  }

  template <typename... tokens>
  void Null(BindingPower bp, NullDenotation nud, Token token, tokens... rest) {
    Null(bp, nud, token);
    Null(bp, nud, rest...);
  }

  void _RegisterLed(LeftBindingPower lbp, RightBindingPower rbp,
                    LeftDenotation led, Token token) {
    null_lookup.try_emplace(token.type, NullInfo(-1));
    left_lookup.emplace(token.type, LeftInfo{lbp, rbp, led});
  }
  template <typename... tokens>
  void _RegisterLed(LeftBindingPower lbp, RightBindingPower rbp,
                    LeftDenotation led, Token token, tokens... rest) {
    _RegisterLed(lbp, rbp, led, token);
    _RegisterLed(lbp, rbp, led, rest...);
  }

  template <typename... tokens>
  void Left(BindingPower bp, LeftDenotation led, tokens... t) {
    _RegisterLed(bp, bp, led, t...);
  }

  template <typename... tokens>
  void LeftRightAssoc(BindingPower bp, LeftDenotation led, tokens... t) {
    _RegisterLed(bp, bp - 1, led, t...);
  }

  NullInfo LookupNull(const Token t) { return null_lookup[t.type]; }
  LeftInfo LookupLeft(const Token t) { return left_lookup[t.type]; }

  std::unordered_map<Token::Type, NullInfo> null_lookup;
  std::unordered_map<Token::Type, LeftInfo> left_lookup;
};

/*
class ParserSpec(object):
"""Specification for a TDOP parser."""

def __init__(self):
self.null_lookup = {}
self.left_lookup = {}

def Null(self, bp, nud, tokens):
"""Register a token that doesn't take anything on the left.

Examples: constant, prefix operator, error.
"""
for token in tokens:
  self.null_lookup[token] = NullInfo(nud=nud, bp=bp)
  if token not in self.left_lookup:
    self.left_lookup[token] = LeftInfo()  # error

def _RegisterLed(self, lbp, rbp, led, tokens):
for token in tokens:
  if token not in self.null_lookup:
    self.null_lookup[token] = NullInfo(NullError)
  self.left_lookup[token] = LeftInfo(lbp=lbp, rbp=rbp, led=led)

def Left(self, bp, led, tokens):
"""Register a token that takes an expression on the left."""
self._RegisterLed(bp, bp, led, tokens)

def LeftRightAssoc(self, bp, led, tokens):
"""Register a right associative operator."""
self._RegisterLed(bp, bp-1, led, tokens)

def LookupNull(self, token):
"""Get the parsing function and precedence for a null position token."""
try:
  nud = self.null_lookup[token]
except KeyError:
  raise ParseError('Unexpected token %r' % token)
return nud

def LookupLeft(self, token):
"""Get the parsing function and precedence for a left position token."""
try:
  led = self.left_lookup[token]
except KeyError:
  raise ParseError('Unexpected token %r' % token)
return led
*/

static Token EOF_TOKEN = Token{Token::Type::Eof};

class Parser {
  class Lexer {
  public:
    Lexer(const char *_text) : text{_text} {
      advance();
      advance();
    }

    Token advance() {
      _current = _peek;
      // Lex out next token.
      if (!text[cursor]) {
        _peek = Token{Token::Type::Eof};
        return _current;
      }

      while (text[cursor] != '\0') {
        while (text[cursor] && isspace(text[cursor])) {
          if (text[cursor] == '\n') {
            cursor++;
          } else {
            cursor++;
          }
        }
        if (!text[cursor]) {
          _peek = Token{Token::Type::Eof};
          return _current;
        }

        switch (text[cursor]) {
        case '(':
          _peek = Token{Token::Type::LeftParen};
          cursor++;
          return current();
        case ')':
          _peek = Token{Token::Type::RightParen};
          cursor++;
          return current();
        case '+':
          if (text[cursor + 1] == '+') {
            _peek = Token{Token::Type::Increment};
            cursor++;
            cursor++;
            return current();
          } else if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::AddAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Plus};
            cursor++;
            return current();
          }
        case '-':
          if (text[cursor + 1] == '-') {
            _peek = Token{Token::Type::Decrement};
            cursor++;
            cursor++;
            return current();
          } else if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::SubAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Sub};
            cursor++;
            return current();
          }

        case '*':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::MulAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Mul};
            cursor++;
            return current();
          }

        case '/':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::DivAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Div};
            cursor++;
            return current();
          }

        case '%':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::ModAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Mod};
            cursor++;
            return current();
          }

        case '!':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::NotEqual};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Not};
            cursor++;
            return current();
          }

        case '=':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::Equal};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Assign};
            cursor++;
            return current();
          }
        case '<':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::LessThanOrEqual};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::LessThan};
            cursor++;
            return current();
          }
        case '>':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::LessThanOrEqual};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::LessThan};
            cursor++;
            return current();
          }
        case '&':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::AndAssign};
            cursor++;
            cursor++;
            return current();
          } else if (text[cursor + 1] == '&') {
            _peek = Token{Token::Type::And};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Ampersand};
            cursor++;
            return current();
          }
        case '|':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::OrAssign};
            cursor++;
            cursor++;
            return current();
          } else if (text[cursor + 1] == '|') {
            _peek = Token{Token::Type::Or};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Pipe};
            cursor++;
            return current();
          }
        case '~':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::TildeAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Tilde};
            cursor++;
            return current();
          }
        default:
          // string or int.
          if (isdigit(text[cursor])) {
            static std::array<char, 1000> buffer;
            // Deal with integer.
            size_t start = cursor;
            while (isdigit(text[cursor++])) {
            }
            size_t end = cursor; // end is one past the last digit.
            if (end - start > buffer.size() - 1) {
              abort(); // Too big of a number.
            }
            size_t j = 0;
            for (size_t i = start; i < end; i++) {
              buffer[j++] = text[i];
            }
            buffer[j] = '\0';
            _peek = Token{Token::Type::BigInt};
            mpz_init(_peek.bigint);
            mpz_set_str(_peek.bigint, buffer.data(), 10);
            return current();
          } else if (isalpha(text[cursor])) {
            size_t start = cursor;
            while (isdigit(text[cursor++])) {
            }
            size_t end = cursor; // end is one past the last digit.
            _peek = Token{Token::Type::String};
            auto temp = std::make_shared<char[]>(end - start + 1);
            memcpy(temp.get(), &text[start], end - start);
            temp[static_cast<ptrdiff_t>(end - start)] = '\0';
            _peek.string = std::move(temp);
            return current();

          } else {
            abort(); // Error, unknown token.
          }
        }
      }

      return _current;
    }

    Token peek() { return _peek; }
    Token current() { return _current; }

    Token _peek{Token::Type::Eof};
    Token _current{Token::Type::Eof};
    const char *text;
    size_t cursor{0};
  } lexer;
  ParserSpec &spec;

public:
  Parser(const char *text, ParserSpec &_spec) : lexer{text}, spec{_spec} {}

  bool AtToken(Token::Type t) { return lexer.current().type == t; }
  void Eat(Token::Type t) {
    if (!AtToken(t)) {
      printf("Unexpected ");
      lexer.current().print();
      abort();
    } else {
      lexer.advance();
    }
  }

  Node ParseUntil(BindingPower rbp) {
    if (AtToken(Token::Type::Eof)) {
      printf("Unexpected end of input!");
      abort();
    }
    auto t = lexer.current();
    lexer.advance();
    auto null_info = spec.LookupNull(t);
    auto node = null_info.nud(*this, t, null_info.bp);

    while (true) {
      t = lexer.current();
      auto left_info = spec.LookupLeft(t);
      if (rbp >= left_info.lbp) {
        break;
      }
      lexer.advance();
      node = left_info.led(*this, t, node, left_info.rbp);
    }
    return node;
  }

  /*
def ParseUntil(self, rbp):
"""
Parse to the right, eating tokens until we encounter a token with binding
power LESS THAN OR EQUAL TO rbp.
"""
if self.AtToken('eof'):
raise ParseError('Unexpected end of input')

  t = self.token
          self.Next()  # skip over the token, e.g. ! ~ + -

      null_info = self.spec.LookupNull(t.type)
            node = null_info.nud(self, t, null_info.bp)

        while True:
      t = self.token
            left_info = self.spec.LookupLeft(t.type)

# Examples:
# If we see 1*2+  , rbp = 27 and lbp = 25, so stop.
# If we see 1+2+  , rbp = 25 and lbp = 25, so stop.
# If we see 1**2**, rbp = 26 and lbp = 27, so keep going.
            if rbp >= left_info.lbp:
      break
      self.Next()  # skip over the token, e.g. / *

      node = left_info.led(self, t, node, left_info.rbp)

        return node
  */
  /*

  EOF_TOKEN = Token('eof', 'eof')


              class Parser(object):
      """Recursive TDOP parser."""

      def __init__(self, spec, lexer):
      self.spec = spec
            self.lexer = lexer  # iterable
        self.token = None  # current token

            def AtToken(self, token_type):
      """Test if we are looking at a token."""
      return self.token.type == token_type

            def Next(self):
      """Move to the next token."""
  try:
      t = self.lexer.__next__()
            except StopIteration:
          t = EOF_TOKEN
                self.token = t

                def Eat(self, val):
          """Assert the value of the current token, then move to the next
  token.""" if val and not self.AtToken(val): raise ParseError('expected %s, got
  %s' % (val, self.token)) self.Next()


          def Parse(self):
          self.Next()
              return self.ParseUntil(0)
                  */

  Node Parse() { return ParseUntil(0); }
};

Node NullConstant(Parser &, Token t, BindingPower) { return Node{t}; }

Node NullParen(Parser &p, Token, BindingPower bp) {
  auto r = p.ParseUntil(bp);
  p.Eat(Token::Type::RightParen);
  return r;
}
/*
#
# Null Denotation -- token that takes nothing on the left
#

def NullConstant(p, token, bp):
  return Node(token)


def NullParen(p, token, bp):
  """ Arithmetic grouping """
  r = p.ParseUntil(bp)
  p.Eat(')')
  return r
*/

Node NullPrefixOp(Parser &p, Token t, BindingPower bp) {
  auto r = p.ParseUntil(bp);
  return Node{t, r};
}

/*
def NullPrefixOp(p, token, bp):
  """Prefix operator.

  Low precedence:  return, raise, etc.
    return x+y is return (x+y), not (return x) + y
  High precedence: logical negation, bitwise complement, etc.
    !x && y is (!x) && y, not !(x && y)
  """
  r = p.ParseUntil(bp)
  return CompositeNode(token, [r])
*/

Node NullIncDec(Parser &p, Token t, BindingPower bp) {
  auto right = p.ParseUntil(bp);
  if (right.type != Node::Type::Leaf && right.t.type != Token::Type::String) {
    abort();
  }

  return Node{t, right};
}
/*
def NullIncDec(p, token, bp):
  """ ++x or ++x[1] """
  right = p.ParseUntil(bp)
  if right.token.type not in ('name', 'get'):
    raise tdop.ParseError("Can't assign to %r (%s)" % (right, right.token))
  return CompositeNode(token, [right])
*/

/*
#
# Left Denotation -- token that takes an expression on the left
#

def LeftIncDec(p, token, left, rbp):
  """ For i++ and i--
  """
  if left.token.type not in ('name', 'get'):
    raise tdop.ParseError("Can't assign to %r (%s)" % (left, left.token))
  token.type = 'post' + token.type
  return CompositeNode(token, [left])


def LeftIndex(p, token, left, unused_bp):
  """ index f[x+1] """
  # f[x] or f[x][y]
  if left.token.type not in ('name', 'get'):
    raise tdop.ParseError("%s can't be indexed" % left)
  index = p.ParseUntil(0)
  p.Eat("]")

  token.type = 'get'
  return CompositeNode(token, [left, index])


def LeftTernary(p, token, left, bp):
  """ e.g. a > 1 ? x : y """
  # 0 binding power since any operators allowed until ':'.  See:
  #
  # http://en.cppreference.com/w/c/language/operator_precedence#cite_note-2
  #
  # "The expression in the middle of the conditional operator (between ? and
  # :) is parsed as if parenthesized: its precedence relative to ?: is
  # ignored."
  true_expr = p.ParseUntil(0)

  p.Eat(':')
  false_expr = p.ParseUntil(bp)
  children = [left, true_expr, false_expr]
  return CompositeNode(token, children)


def LeftBinaryOp(p, token, left, rbp):
  """ Normal binary operator like 1+2 or 2*3, etc. """
  return CompositeNode(token, [left, p.ParseUntil(rbp)])


def LeftAssign(p, token, left, rbp):
  """ Normal binary operator like 1+2 or 2*3, etc. """
  # x += 1, or a[i] += 1
  if left.token.type not in ('name', 'get'):
    raise tdop.ParseError("Can't assign to %r (%s)" % (left, left.token))
  return CompositeNode(token, [left, p.ParseUntil(rbp)])


def LeftComma(p, token, left, rbp):
  """ foo, bar, baz
  Could be sequencing operator, or tuple without parens
  """
  r = p.ParseUntil(rbp)
  if left.token.type == ',':  # Keep adding more children
    left.children.append(r)
    return left
  children = [left, r]
  return CompositeNode(token, children)


# For overloading of , inside function calls
COMMA_PREC = 1

def LeftFuncCall(p, token, left, unused_bp):
  """ Function call f(a, b). """
  children = [left]
  # f(x) or f[i](x)
  if left.token.type not in ('name', 'get'):
    raise tdop.ParseError("%s can't be called" % left)
  while not p.AtToken(')'):
    # We don't want to grab the comma, e.g. it is NOT a sequence operator.  So
    # set the precedence to 5.
    children.append(p.ParseUntil(COMMA_PREC))
    if p.AtToken(','):
      p.Next()
  p.Eat(")")
  token.type = 'call'
  return CompositeNode(token, children)


def MakeShellParserSpec():
  """
  Create a parser.
  Compare the code below with this table of C operator precedence:
  http://en.cppreference.com/w/c/language/operator_precedence
  """
  spec = tdop.ParserSpec()

  spec.Left(31, LeftIncDec, ['++', '--'])
  spec.Left(31, LeftFuncCall, ['('])
  spec.Left(31, LeftIndex, ['['])

  # 29 -- binds to everything except function call, indexing, postfix ops
  spec.Null(29, NullIncDec, ['++', '--'])

  # Right associative: 2 ** 3 ** 2 == 2 ** (3 ** 2)
  # Binds more strongly than negation.
  spec.LeftRightAssoc(29, LeftBinaryOp, ['**'])

  spec.Null(27, NullPrefixOp, ['+', '!', '~', '-'])

  spec.Left(25, LeftBinaryOp, ['*', '/', '%'])

  spec.Left(23, LeftBinaryOp, ['+', '-'])
  spec.Left(21, LeftBinaryOp, ['<<', '>>'])
  spec.Left(19, LeftBinaryOp, ['<', '>', '<=', '>='])
  spec.Left(17, LeftBinaryOp, ['!=', '=='])

  spec.Left(15, LeftBinaryOp, ['&'])
  spec.Left(13, LeftBinaryOp, ['^'])
  spec.Left(11, LeftBinaryOp, ['|'])
  spec.Left(9, LeftBinaryOp, ['&&'])
  spec.Left(7, LeftBinaryOp, ['||'])

  spec.LeftRightAssoc(5, LeftTernary, ['?'])

  # Right associative: a = b = 2 is a = (b = 2)
  spec.LeftRightAssoc(3, LeftAssign, [
      '=',
      '+=', '-=', '*=', '/=', '%=',
      '<<=', '>>=', '&=', '^=', '|='])

  spec.Left(COMMA_PREC, LeftComma, [','])

  # 0 precedence -- doesn't bind until )
  spec.Null(0, NullParen, ['('])  # for grouping

  # -1 precedence -- never used
  spec.Null(-1, NullConstant, ['name', 'number'])
  spec.Null(-1, tdop.NullError, [')', ']', ':', 'eof'])

  return spec


def MakeParser(s):
  """Used by tests."""
  spec = MakeShellParserSpec()
  lexer = tdop.Tokenize(s)
  p = tdop.Parser(spec, lexer)
  return p


def ParseShell(s, expected=None):
  """Used by tests."""
  p = MakeParser(s)
  tree = p.Parse()

  sexpr = repr(tree)
  if expected is not None:
    assert sexpr == expected, '%r != %r' % (sexpr, expected)

  print('%-40s %s' % (s, sexpr))
  return tree


def main(argv):
  try:
    s = argv[1]
  except IndexError:
    print('Usage: ./arith_parse.py EXPRESSION')
  else:
    try:
      tree = ParseShell(s)
    except tdop.ParseError as e:
      print('Error parsing %r: %s' % (s, e), file=sys.stderr)


*/

const char *parse(const char *in) { return in; }

int main() {
  assert(!strcmp(parse("1+2*3"), "(+ 1 (* 2 3))"));

  return 0;
}
