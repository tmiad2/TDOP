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


#define TokenTypes                                                             \
  X(Eof, "EOF")                                                                \
  X(Get, "Get")                                                                \
  X(Call, "Call")                                                              \
  X(LeftShift, "<<")                                                           \
  X(RightShift, ">>")                                                          \
  X(LeftShiftAssign, "<<=")                                                    \
  X(RightShiftAssign, ">>=")                                                   \
  X(QuestionMark, "?")                                                         \
  X(Caret, "^")                                                                \
  X(XorAssign, "^=")                                                           \
  X(Comma, ",")                                                                \
  X(Integer, "Integer")                                                        \
  X(String, "Id")                                                              \
  X(Plus, "+")                                                                 \
  X(Sub, "-")                                                                  \
  X(Mul, "*")                                                                  \
  X(Div, "/")                                                                  \
  X(Increment, "++")                                                           \
  X(Decrement, "--")                                                           \
  X(AddAssign, "+=")                                                           \
  X(MulAssign, "*=")                                                           \
  X(SubAssign, "-=")                                                           \
  X(DivAssign, "/=")                                                           \
  X(ModAssign, "%=")                                                           \
  X(AndAssign, "&=")                                                           \
  X(Assign, "=")                                                               \
  X(OrAssign, "|=")                                                            \
  X(Tilde, "~")                                                                \
  X(TildeAssign, "~=")                                                         \
  X(Equal, "==")                                                               \
  X(NotEqual, "!=")                                                            \
  X(GreaterThan, ">")                                                          \
  X(LessThan, "<")                                                             \
  X(GreaterThanOrEqual, ">=")                                                  \
  X(LessThanOrEqual, "<=")                                                     \
  X(Mod, "%")                                                                  \
  X(Ampersand, "&")                                                            \
  X(And, "&&")                                                                 \
  X(Colon, ":")                                                                \
  X(Pipe, "|")                                                                 \
  X(Or, "||")                                                                  \
  X(Not, "!")                                                                  \
  X(LeftParen, "(")                                                            \
  X(RightParen, ")")                                                           \
  X(RightSquareBracket, "]")                                                   \
  X(LeftSquareBracket, "[")                                                    \
  X(RightCurlyBrace, "}")                                                      \
  X(LeftCurlyBrace, "{")                                                       \
  X(BigInt, "BigInt")

struct Token {
  constexpr Token(const Token &other) { *this = other; }
  constexpr Token &operator=(const Token &other) {
    type = other.type;
    switch (other.type) {
    case Type::String:
      break;
    case Type::BigInt:
      mpz_init(bigint);
      mpz_set(bigint, other.bigint);
      break;
    case Type::Integer:
      value = other.value;
      break;
    default:
      break;
    }
    return *this;
  }
  constexpr Token(Token &&other) { *this = std::move(other); }
  constexpr Token &operator=(Token &&other) {
    type = other.type;
    switch (other.type) {
    case Type::String:
      break;
    case Type::BigInt:
      mpz_init(bigint);
      mpz_swap(bigint, other.bigint);
      break;
    case Type::Integer:
      value = other.value;
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
#define X(s, n) s,
    TokenTypes
  } type;

  Token(const std::shared_ptr<char[]> _string)
      : string{_string}, type{Type::String} {}
  constexpr Token(const Token::Type t) : value{0}, type{t} {}

  bool operator==(Token const &o) const {
    return type == o.type && value == o.value;
  }

  void print() const {
    switch (type) {
#define X(s, n)                                                                \
  case Type::s:                                                                \
    printf("%s", n);                                                           \
    break;
      TokenTypes
    }
    if (type == Type::String) {
      printf(": %s", string.get());
    } else if (type == Type::BigInt) {
      printf(": ");
      mpz_out_raw(stdout, bigint);
    }
  }
};


struct Node {
  class NodeTree {
  public:
    NodeTree() = default;
    NodeTree &operator=(const NodeTree &n) {
      if (&n == this)
        return *this;
      N1 = n.N1;
      return *this;
    }

    NodeTree &operator=(NodeTree &&n) = default;

    NodeTree(const NodeTree &n) = default;

    template <class... Args> NodeTree(Args &&...args) {
      N1.reserve(sizeof...(args));
      add(args...);
    }

    std::vector<struct Node> N1{}; // array of operators, length long.
    struct Node &operator[](size_t i) {
      if (i < N1.size()) {
        return N1[i];
      } else {
        abort();
      }
    }
    const struct Node &operator[](size_t i) const {
      if (i < N1.size()) {
        return N1[i];
      } else {
        abort();
      }
    }
    size_t size() const { return N1.size(); }

  private:
    void add() {}
    void add(Token &in) { N1.emplace_back(in); }
    void add(NodeTree &in) { N1.push_back(in); }
    void add(Node &in) { N1.emplace_back(in); }

    template <class T, class... Args> void add(T &first, Args &&...args) {
      N1.emplace_back(first);
      if (sizeof...(args) > 0) {
        add(args...);
      }
    }
  };
  explicit Node(struct Token t) : t{t} { type = Type::Leaf; }

  Node(const NodeTree &n) {
    N1 = n;
    type = Type::Tree;
  }

  Node() {
    type = Type::Leaf;
    t = Token{Token::Type::Eof};
  }
  Node(const Node &n) = delete;

  template <typename... args>
  Node(struct Token t, args... n) : N1{NodeTree(t, n...)}, type{Type::Tree} {}

  Node(struct Token t, const Node &n) {
    type = Type::Tree;
    N1 = NodeTree(t, n);
  }

  Node &operator=(const Node &n) = delete;
  Node &operator=(Node &&n) {
    type = n.type;
    if (type == Type::Leaf) {
      t = n.t;
    } else {
      N1 = n.N1;
    }
    return *this;
  }

  ~Node() {}

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
      for (size_t i = 0; i < N1.size(); i++) {
        N1[i].print();
      }
      printf(")");
    }
  }
};

class BindingPower {
public:
  BindingPower(int _bp) : bp{_bp} {}
  int bp;
  operator int() const { return bp; }
};

class Parser;
using NullDenotation =
    std::function<Node(Parser &, struct Token, BindingPower)>;
using LeftDenotation =
    std::function<Node(Parser &, struct Token, Node &, BindingPower)>;

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
  LeftInfo(BindingPower _lbp, BindingPower _rbp,
           LeftDenotation _led = LeftError)
      : led{_led}, lbp{_lbp}, rbp{_rbp} {}
  LeftDenotation led{LeftError};
  BindingPower lbp{-1};
  BindingPower rbp{-1};

  static LeftDenotation LeftError;
};
LeftDenotation LeftInfo::LeftError = [](Parser &, struct Token t, Node &,
                                        BindingPower) {
  t.print();
  printf(" cannot be used in infix position!\n");
  abort();
  return Node{t};
};

class ParserSpec {
public:
  ParserSpec() = default;
  ParserSpec(ParserSpec &&) = default;

  void Null(BindingPower bp, NullDenotation nud, Token::Type token) {
    null_lookup[token] = NullInfo{bp, nud};
    left_lookup.try_emplace(token, LeftInfo(-1, -1));
  }

  template <typename... tokens>
  void Null(BindingPower bp, NullDenotation nud, Token::Type token,
            tokens... rest) {
    Null(bp, nud, token);
    Null(bp, nud, rest...);
  }

  void _RegisterLed(BindingPower lbp, BindingPower rbp, LeftDenotation led,
                    Token::Type token) {
    null_lookup.try_emplace(token, NullInfo(-1));
    left_lookup[token] = LeftInfo{lbp, rbp, led};
  }
  template <typename... tokens>
  void _RegisterLed(BindingPower lbp, BindingPower rbp, LeftDenotation led,
                    Token::Type token, tokens... rest) {
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

static Token EOF_TOKEN = Token{Token::Type::Eof};

class Parser {
  class Lexer {
  public:
    Lexer(const char *_text) : text{_text} {
      advance();
      advance();
    }

    Token advance() {
      _current = std::move(_peek);
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
        case ',':
          _peek = Token{Token::Type::Comma};
          cursor++;
          return current();
        case ':':
          _peek = Token{Token::Type::Colon};
          cursor++;
          return current();
        case '(':
          _peek = Token{Token::Type::LeftParen};
          cursor++;
          return current();
        case ')':
          _peek = Token{Token::Type::RightParen};
          cursor++;
          return current();
        case '[':
          _peek = Token{Token::Type::LeftSquareBracket};
          cursor++;
          return current();
        case ']':
          _peek = Token{Token::Type::RightSquareBracket};
          cursor++;
          return current();
        case '{':
          _peek = Token{Token::Type::LeftCurlyBrace};
          cursor++;
          return current();
        case '}':
          _peek = Token{Token::Type::RightCurlyBrace};
          cursor++;
          return current();
        case '?':
          _peek = Token{Token::Type::QuestionMark};
          cursor++;
          return current();

        case '^':
          if (text[cursor + 1] == '=') {
            _peek = Token{Token::Type::XorAssign};
            cursor++;
            cursor++;
            return current();
          } else {
            _peek = Token{Token::Type::Caret};
            cursor++;
            return current();
          }

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
          } else if (text[cursor + 1] == '<') {
            if (text[cursor + 2] == '=') {
              _peek = Token{Token::Type::LeftShiftAssign};
              cursor++;
              cursor++;
              cursor++;
              return current();
            } else {
              _peek = Token{Token::Type::LeftShift};
              cursor++;
              cursor++;
              return current();
            }
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
          } else if (text[cursor + 1] == '>') {
            if (text[cursor + 2] == '=') {
              _peek = Token{Token::Type::RightShiftAssign};
              cursor++;
              cursor++;
              cursor++;
              return current();
            } else {
              _peek = Token{Token::Type::RightShift};
              cursor++;
              cursor++;
              return current();
            }
          }

          else {
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
            cursor--;
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

  void Next() { lexer.advance(); }

  Node Parse() { return ParseUntil(0); }
};

Node NullConstant(Parser &, Token t, BindingPower) { return Node{t}; }

Node NullParen(Parser &p, Token, BindingPower bp) {
  auto r = p.ParseUntil(bp);
  p.Eat(Token::Type::RightParen);
  return r;
}


Node NullPrefixOp(Parser &p, Token t, BindingPower bp) {
  auto r = p.ParseUntil(bp);
  return Node{t, r};
}

Node NullIncDec(Parser &p, Token t, BindingPower bp) {
  auto right = p.ParseUntil(bp);
  if (right.type != Node::Type::Leaf && (right.t.type != Token::Type::String &&
                                         right.t.type != Token::Type::Get)) {
    abort();
  }

  return Node{t, right};
}

Node LeftIncDec(Parser &, Token t, Node left, BindingPower) {
  if (left.type != Node::Type::Leaf &&
      (left.t.type != Token::Type::String && left.t.type != Token::Type::Get)) {
    abort();
  }
  // todo prepend post to token type.
  return Node{t, left};
}

Node LeftIndex(Parser &p, Token t, Node left, BindingPower) {
  if (left.type != Node::Type::Leaf &&
      (left.t.type != Token::Type::String && left.t.type != Token::Type::Get)) {
    abort();
  }
  auto index = p.ParseUntil(0);
  p.Eat(Token::Type::RightSquareBracket);
  t.type = Token::Type::Get;
  return Node{t, left, index};
}

Node LeftTernary(Parser &p, Token t, Node left, BindingPower bp) {
  auto true_expr = p.ParseUntil(0);
  p.Eat(Token::Type::Colon);
  auto false_expr = p.ParseUntil(bp);
  return Node{t, left, true_expr, false_expr};
}


Node LeftBinaryOp(Parser &p, Token t, Node left, BindingPower bp) {
  return Node{t, left, p.ParseUntil(bp)};
}


Node LeftAssign(Parser &p, Token t, Node left, BindingPower bp) {
  if (left.type != Node::Type::Leaf &&
      (left.t.type != Token::Type::String && left.t.type != Token::Type::Get)) {
    abort();
  }
  return Node{t, left, p.ParseUntil(bp)};
}

Node LeftComma(Parser &p, Token t, Node left, BindingPower bp) {
  auto r = p.ParseUntil(bp);
  if (left.type == Node::Type::Leaf && left.t.type == Token::Type::Comma) {
    left.N1.N1.emplace_back(r);
    return left;
  }
  return Node{t, left, r};
}

static constexpr int COMMA_PREC = 1;
Node LeftFuncCall(Parser &p, Token t, Node left, BindingPower bp) {

  if (left.type != Node::Type::Leaf &&
      (left.t.type != Token::Type::String && left.t.type != Token::Type::Get)) {
    abort();
  }
  Node::NodeTree N{};
  N.N1.emplace_back(left);
  while (!p.AtToken(Token::Type::RightParen)) {
    N.N1.emplace_back(p.ParseUntil(COMMA_PREC));
    if (p.AtToken(Token::Type::Comma)) {
      p.Next();
    }
  }
  p.Eat(Token::Type::RightParen);
  t.type = Token::Type::Call;
  return Node{t, N};
}

ParserSpec MakeParserSpec() {

  ParserSpec spec;
  spec.Left(31, LeftIncDec, Token::Type::Increment, Token::Type::Decrement);
  spec.Left(31, LeftFuncCall, Token::Type::LeftParen);
  spec.Left(31, LeftIndex, Token::Type::LeftSquareBracket);

  spec.Null(29, NullIncDec, Token::Type::Increment, Token::Type::Decrement);

  spec.Null(27, NullPrefixOp, Token::Type::Plus, Token::Type::Not,
            Token::Type::Tilde, Token::Type::Sub);

  spec.Left(25, LeftBinaryOp, Token::Type::Mul, Token::Type::Div,
            Token::Type::Mod);

  spec.Left(23, LeftBinaryOp, Token::Type::Plus, Token::Type::Sub);
  spec.Left(21, LeftBinaryOp, Token::Type::LeftShift, Token::Type::RightShift);
  spec.Left(19, LeftBinaryOp, Token::Type::LessThan, Token::Type::GreaterThan,
            Token::Type::LessThanOrEqual, Token::Type::GreaterThanOrEqual);
  spec.Left(17, LeftBinaryOp, Token::Type::NotEqual, Token::Type::Equal);

  spec.Left(15, LeftBinaryOp, Token::Type::Ampersand);
  spec.Left(13, LeftBinaryOp, Token::Type::Caret);
  spec.Left(11, LeftBinaryOp, Token::Type::Pipe);
  spec.Left(9, LeftBinaryOp, Token::Type::And);
  spec.Left(7, LeftBinaryOp, Token::Type::Or);

  spec.LeftRightAssoc(5, LeftTernary, Token::Type::QuestionMark);

  spec.LeftRightAssoc(3, LeftAssign, Token::Type::Assign,
                      Token::Type::AddAssign, Token::Type::SubAssign,
                      Token::Type::MulAssign, Token::Type::DivAssign,
                      Token::Type::ModAssign, Token::Type::LeftShiftAssign,
                      Token::Type::RightShiftAssign, Token::Type::AndAssign,
                      Token::Type::OrAssign, Token::Type::XorAssign);

  spec.Left(COMMA_PREC, LeftComma, Token::Type::Comma);

  spec.Null(0, NullParen, Token::Type::LeftParen); //  # for grouping

  spec.Null(-1, NullConstant, Token::Type::BigInt, Token::Type::Integer,
            Token::Type::String);
  spec.Null(-1, NullInfo::NullError, Token::Type::RightParen,
            Token::Type::RightSquareBracket, Token::Type::RightCurlyBrace,
            Token::Type::Colon, Token::Type::Eof);

  return spec;
}


const char *parse(const char *in) { return in; }

int main() {
  auto ps = MakeParserSpec();
  Parser p{"1+2*3", ps};

  auto N = p.Parse();
  N.print();

  // assert(!strcmp(parse("1+2*3"), "(+ 1 (* 2 3))"));

  return 0;
}
