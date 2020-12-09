# Generated from /home/lukas/Workspace/prcvut/pyneuralogic/neuralogic/grammar/Neuralogic.g4 by ANTLR 4.8
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys

if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\34")
        buf.write("\u0114\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r\4\16")
        buf.write("\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23\t\23")
        buf.write("\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31")
        buf.write("\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36")
        buf.write("\4\37\t\37\3\2\7\2@\n\2\f\2\16\2C\13\2\3\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\5\3N\n\3\3\4\3\4\3\4\3\4\6\4T\n\4")
        buf.write("\r\4\16\4U\3\4\6\4Y\n\4\r\4\16\4Z\5\4]\n\4\3\5\3\5\6\5")
        buf.write("a\n\5\r\5\16\5b\3\5\3\5\3\6\3\6\3\7\3\7\3\7\3\7\3\7\6")
        buf.write("\7n\n\7\r\7\16\7o\3\7\3\7\3\7\6\7u\n\7\r\7\16\7v\5\7y")
        buf.write("\n\7\3\b\3\b\3\b\3\t\5\t\177\n\t\3\t\5\t\u0082\n\t\3\t")
        buf.write("\3\t\5\t\u0086\n\t\3\n\3\n\3\n\3\n\7\n\u008c\n\n\f\n\16")
        buf.write("\n\u008f\13\n\5\n\u0091\n\n\3\n\3\n\3\13\3\13\5\13\u0097")
        buf.write("\n\13\3\f\3\f\3\r\3\r\3\16\5\16\u009e\n\16\3\16\5\16\u00a1")
        buf.write("\n\16\3\16\3\16\3\16\5\16\u00a6\n\16\3\17\3\17\3\17\7")
        buf.write("\17\u00ab\n\17\f\17\16\17\u00ae\13\17\3\20\3\20\3\20\3")
        buf.write("\20\5\20\u00b4\n\20\3\20\5\20\u00b7\n\20\3\21\3\21\3\21")
        buf.write("\3\21\7\21\u00bd\n\21\f\21\16\21\u00c0\13\21\5\21\u00c2")
        buf.write("\n\21\3\21\3\21\3\22\3\22\3\22\3\22\3\22\5\22\u00cb\n")
        buf.write("\22\3\22\3\22\5\22\u00cf\n\22\3\23\3\23\3\23\3\24\3\24")
        buf.write("\3\24\3\25\3\25\3\25\3\25\3\26\3\26\3\27\3\27\3\27\5\27")
        buf.write("\u00e0\n\27\3\27\3\27\5\27\u00e4\n\27\3\30\3\30\3\30\3")
        buf.write("\30\3\31\3\31\3\32\3\32\3\32\3\32\5\32\u00f0\n\32\3\33")
        buf.write("\3\33\3\34\3\34\3\34\3\34\7\34\u00f8\n\34\f\34\16\34\u00fb")
        buf.write("\13\34\3\34\3\34\3\35\3\35\6\35\u0101\n\35\r\35\16\35")
        buf.write("\u0102\3\35\3\35\3\36\3\36\3\36\3\36\7\36\u010b\n\36\f")
        buf.write("\36\16\36\u010e\13\36\3\36\3\36\3\37\3\37\3\37\2\2 \2")
        buf.write('\4\6\b\n\f\16\20\22\24\26\30\32\34\36 "$&(*,.\60\62\64')
        buf.write("\668:<\2\4\3\2\5\7\3\2\5\6\2\u011c\2A\3\2\2\2\4M\3\2\2")
        buf.write("\2\6\\\3\2\2\2\b`\3\2\2\2\nf\3\2\2\2\fx\3\2\2\2\16z\3")
        buf.write("\2\2\2\20~\3\2\2\2\22\u0087\3\2\2\2\24\u0096\3\2\2\2\26")
        buf.write("\u0098\3\2\2\2\30\u009a\3\2\2\2\32\u009d\3\2\2\2\34\u00a7")
        buf.write('\3\2\2\2\36\u00af\3\2\2\2 \u00b8\3\2\2\2"\u00c5\3\2\2')
        buf.write("\2$\u00d0\3\2\2\2&\u00d3\3\2\2\2(\u00d6\3\2\2\2*\u00da")
        buf.write("\3\2\2\2,\u00df\3\2\2\2.\u00e5\3\2\2\2\60\u00e9\3\2\2")
        buf.write("\2\62\u00ef\3\2\2\2\64\u00f1\3\2\2\2\66\u00f3\3\2\2\2")
        buf.write("8\u00fe\3\2\2\2:\u0106\3\2\2\2<\u0111\3\2\2\2>@\5\4\3")
        buf.write("\2?>\3\2\2\2@C\3\2\2\2A?\3\2\2\2AB\3\2\2\2B\3\3\2\2\2")
        buf.write('CA\3\2\2\2DN\5"\22\2EN\5\16\b\2FG\5\34\17\2GH\7\3\2\2')
        buf.write("HN\3\2\2\2IN\5&\24\2JN\5$\23\2KN\5(\25\2LN\5*\26\2MD\3")
        buf.write("\2\2\2ME\3\2\2\2MF\3\2\2\2MI\3\2\2\2MJ\3\2\2\2MK\3\2\2")
        buf.write("\2ML\3\2\2\2N\5\3\2\2\2OP\5\n\6\2PQ\7\b\2\2QR\5\b\5\2")
        buf.write("RT\3\2\2\2SO\3\2\2\2TU\3\2\2\2US\3\2\2\2UV\3\2\2\2V]\3")
        buf.write("\2\2\2WY\5\b\5\2XW\3\2\2\2YZ\3\2\2\2ZX\3\2\2\2Z[\3\2\2")
        buf.write('\2[]\3\2\2\2\\S\3\2\2\2\\X\3\2\2\2]\7\3\2\2\2^a\5"\22')
        buf.write("\2_a\5\34\17\2`^\3\2\2\2`_\3\2\2\2ab\3\2\2\2b`\3\2\2\2")
        buf.write("bc\3\2\2\2cd\3\2\2\2de\7\3\2\2e\t\3\2\2\2fg\5\34\17\2")
        buf.write("g\13\3\2\2\2hi\5\20\t\2ij\7\b\2\2jk\5\34\17\2kl\7\3\2")
        buf.write("\2ln\3\2\2\2mh\3\2\2\2no\3\2\2\2om\3\2\2\2op\3\2\2\2p")
        buf.write("y\3\2\2\2qr\5\34\17\2rs\7\3\2\2su\3\2\2\2tq\3\2\2\2uv")
        buf.write("\3\2\2\2vt\3\2\2\2vw\3\2\2\2wy\3\2\2\2xm\3\2\2\2xt\3\2")
        buf.write("\2\2y\r\3\2\2\2z{\5\20\t\2{|\7\3\2\2|\17\3\2\2\2}\177")
        buf.write("\5,\27\2~}\3\2\2\2~\177\3\2\2\2\177\u0081\3\2\2\2\u0080")
        buf.write("\u0082\5<\37\2\u0081\u0080\3\2\2\2\u0081\u0082\3\2\2\2")
        buf.write("\u0082\u0083\3\2\2\2\u0083\u0085\5\32\16\2\u0084\u0086")
        buf.write("\5\22\n\2\u0085\u0084\3\2\2\2\u0085\u0086\3\2\2\2\u0086")
        buf.write("\21\3\2\2\2\u0087\u0090\7\20\2\2\u0088\u008d\5\24\13\2")
        buf.write("\u0089\u008a\7\22\2\2\u008a\u008c\5\24\13\2\u008b\u0089")
        buf.write("\3\2\2\2\u008c\u008f\3\2\2\2\u008d\u008b\3\2\2\2\u008d")
        buf.write("\u008e\3\2\2\2\u008e\u0091\3\2\2\2\u008f\u008d\3\2\2\2")
        buf.write("\u0090\u0088\3\2\2\2\u0090\u0091\3\2\2\2\u0091\u0092\3")
        buf.write("\2\2\2\u0092\u0093\7\21\2\2\u0093\23\3\2\2\2\u0094\u0097")
        buf.write("\5\30\r\2\u0095\u0097\5\26\f\2\u0096\u0094\3\2\2\2\u0096")
        buf.write("\u0095\3\2\2\2\u0097\25\3\2\2\2\u0098\u0099\7\4\2\2\u0099")
        buf.write("\27\3\2\2\2\u009a\u009b\t\2\2\2\u009b\31\3\2\2\2\u009c")
        buf.write("\u009e\7\31\2\2\u009d\u009c\3\2\2\2\u009d\u009e\3\2\2")
        buf.write("\2\u009e\u00a0\3\2\2\2\u009f\u00a1\7\30\2\2\u00a0\u009f")
        buf.write("\3\2\2\2\u00a0\u00a1\3\2\2\2\u00a1\u00a2\3\2\2\2\u00a2")
        buf.write("\u00a5\7\7\2\2\u00a3\u00a4\7\23\2\2\u00a4\u00a6\7\5\2")
        buf.write("\2\u00a5\u00a3\3\2\2\2\u00a5\u00a6\3\2\2\2\u00a6\33\3")
        buf.write("\2\2\2\u00a7\u00ac\5\20\t\2\u00a8\u00a9\7\22\2\2\u00a9")
        buf.write("\u00ab\5\20\t\2\u00aa\u00a8\3\2\2\2\u00ab\u00ae\3\2\2")
        buf.write("\2\u00ac\u00aa\3\2\2\2\u00ac\u00ad\3\2\2\2\u00ad\35\3")
        buf.write("\2\2\2\u00ae\u00ac\3\2\2\2\u00af\u00b0\7\7\2\2\u00b0\u00b6")
        buf.write("\7\t\2\2\u00b1\u00b7\5\62\32\2\u00b2\u00b4\7\26\2\2\u00b3")
        buf.write("\u00b2\3\2\2\2\u00b3\u00b4\3\2\2\2\u00b4\u00b5\3\2\2\2")
        buf.write("\u00b5\u00b7\7\7\2\2\u00b6\u00b1\3\2\2\2\u00b6\u00b3\3")
        buf.write("\2\2\2\u00b7\37\3\2\2\2\u00b8\u00c1\7\16\2\2\u00b9\u00be")
        buf.write("\5\36\20\2\u00ba\u00bb\7\22\2\2\u00bb\u00bd\5\36\20\2")
        buf.write("\u00bc\u00ba\3\2\2\2\u00bd\u00c0\3\2\2\2\u00be\u00bc\3")
        buf.write("\2\2\2\u00be\u00bf\3\2\2\2\u00bf\u00c2\3\2\2\2\u00c0\u00be")
        buf.write("\3\2\2\2\u00c1\u00b9\3\2\2\2\u00c1\u00c2\3\2\2\2\u00c2")
        buf.write("\u00c3\3\2\2\2\u00c3\u00c4\7\17\2\2\u00c4!\3\2\2\2\u00c5")
        buf.write("\u00c6\5\20\t\2\u00c6\u00c7\7\b\2\2\u00c7\u00ca\5\34\17")
        buf.write("\2\u00c8\u00c9\7\22\2\2\u00c9\u00cb\5\60\31\2\u00ca\u00c8")
        buf.write("\3\2\2\2\u00ca\u00cb\3\2\2\2\u00cb\u00cc\3\2\2\2\u00cc")
        buf.write("\u00ce\7\3\2\2\u00cd\u00cf\5 \21\2\u00ce\u00cd\3\2\2\2")
        buf.write("\u00ce\u00cf\3\2\2\2\u00cf#\3\2\2\2\u00d0\u00d1\5\32\16")
        buf.write("\2\u00d1\u00d2\5,\27\2\u00d2%\3\2\2\2\u00d3\u00d4\5\32")
        buf.write("\16\2\u00d4\u00d5\5 \21\2\u00d5'\3\2\2\2\u00d6\u00d7")
        buf.write("\7\26\2\2\u00d7\u00d8\7\7\2\2\u00d8\u00d9\5 \21\2\u00d9")
        buf.write(")\3\2\2\2\u00da\u00db\5 \21\2\u00db+\3\2\2\2\u00dc\u00dd")
        buf.write("\7\26\2\2\u00dd\u00de\7\7\2\2\u00de\u00e0\7\t\2\2\u00df")
        buf.write("\u00dc\3\2\2\2\u00df\u00e0\3\2\2\2\u00e0\u00e3\3\2\2\2")
        buf.write("\u00e1\u00e4\5.\30\2\u00e2\u00e4\5\62\32\2\u00e3\u00e1")
        buf.write("\3\2\2\2\u00e3\u00e2\3\2\2\2\u00e4-\3\2\2\2\u00e5\u00e6")
        buf.write("\7\f\2\2\u00e6\u00e7\5\62\32\2\u00e7\u00e8\7\r\2\2\u00e8")
        buf.write("/\3\2\2\2\u00e9\u00ea\5,\27\2\u00ea\61\3\2\2\2\u00eb\u00f0")
        buf.write("\5\64\33\2\u00ec\u00f0\5\66\34\2\u00ed\u00f0\58\35\2\u00ee")
        buf.write("\u00f0\5:\36\2\u00ef\u00eb\3\2\2\2\u00ef\u00ec\3\2\2\2")
        buf.write("\u00ef\u00ed\3\2\2\2\u00ef\u00ee\3\2\2\2\u00f0\63\3\2")
        buf.write("\2\2\u00f1\u00f2\t\3\2\2\u00f2\65\3\2\2\2\u00f3\u00f4")
        buf.write("\7\16\2\2\u00f4\u00f9\5\64\33\2\u00f5\u00f6\7\22\2\2\u00f6")
        buf.write("\u00f8\5\64\33\2\u00f7\u00f5\3\2\2\2\u00f8\u00fb\3\2\2")
        buf.write("\2\u00f9\u00f7\3\2\2\2\u00f9\u00fa\3\2\2\2\u00fa\u00fc")
        buf.write("\3\2\2\2\u00fb\u00f9\3\2\2\2\u00fc\u00fd\7\17\2\2\u00fd")
        buf.write("\67\3\2\2\2\u00fe\u0100\7\16\2\2\u00ff\u0101\5\66\34\2")
        buf.write("\u0100\u00ff\3\2\2\2\u0101\u0102\3\2\2\2\u0102\u0100\3")
        buf.write("\2\2\2\u0102\u0103\3\2\2\2\u0103\u0104\3\2\2\2\u0104\u0105")
        buf.write("\7\17\2\2\u01059\3\2\2\2\u0106\u0107\7\n\2\2\u0107\u010c")
        buf.write("\5\64\33\2\u0108\u0109\7\22\2\2\u0109\u010b\5\64\33\2")
        buf.write("\u010a\u0108\3\2\2\2\u010b\u010e\3\2\2\2\u010c\u010a\3")
        buf.write("\2\2\2\u010c\u010d\3\2\2\2\u010d\u010f\3\2\2\2\u010e\u010c")
        buf.write("\3\2\2\2\u010f\u0110\7\13\2\2\u0110;\3\2\2\2\u0111\u0112")
        buf.write('\7\27\2\2\u0112=\3\2\2\2"AMUZ\\`bovx~\u0081\u0085\u008d')
        buf.write("\u0090\u0096\u009d\u00a0\u00a5\u00ac\u00b3\u00b6\u00be")
        buf.write("\u00c1\u00ca\u00ce\u00df\u00e3\u00ef\u00f9\u0102\u010c")
        return buf.getvalue()


class NeuralogicParser(Parser):

    grammarFileName = "Neuralogic.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]

    sharedContextCache = PredictionContextCache()

    literalNames = [
        "<INVALID>",
        "'.'",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "':-'",
        "'='",
        "'{'",
        "'}'",
        "'<'",
        "'>'",
        "'['",
        "']'",
        "'('",
        "')'",
        "','",
        "'/'",
        "'^'",
        "'true'",
        "'$'",
        "'~'",
        "'@'",
        "'*'",
    ]

    symbolicNames = [
        "<INVALID>",
        "<INVALID>",
        "VARIABLE",
        "INT",
        "FLOAT",
        "ATOMIC_NAME",
        "IMPLIED_BY",
        "ASSIGN",
        "LCURL",
        "RCURL",
        "LANGLE",
        "RANGLE",
        "LBRACKET",
        "RBRACKET",
        "LPAREN",
        "RPAREN",
        "COMMA",
        "SLASH",
        "CARET",
        "TRUE",
        "DOLLAR",
        "NEGATION",
        "SPECIAL",
        "PRIVATE",
        "WS",
        "COMMENT",
        "MULTILINE_COMMENT",
    ]

    RULE_templateFile = 0
    RULE_templateLine = 1
    RULE_examplesFile = 2
    RULE_liftedExample = 3
    RULE_label = 4
    RULE_queriesFile = 5
    RULE_fact = 6
    RULE_atom = 7
    RULE_termList = 8
    RULE_term = 9
    RULE_variable = 10
    RULE_constant = 11
    RULE_predicate = 12
    RULE_conjunction = 13
    RULE_metadataVal = 14
    RULE_metadataList = 15
    RULE_lrnnRule = 16
    RULE_predicateOffset = 17
    RULE_predicateMetadata = 18
    RULE_weightMetadata = 19
    RULE_templateMetadata = 20
    RULE_weight = 21
    RULE_fixedValue = 22
    RULE_offset = 23
    RULE_value = 24
    RULE_number = 25
    RULE_vector = 26
    RULE_matrix = 27
    RULE_dimensions = 28
    RULE_negation = 29

    ruleNames = [
        "templateFile",
        "templateLine",
        "examplesFile",
        "liftedExample",
        "label",
        "queriesFile",
        "fact",
        "atom",
        "termList",
        "term",
        "variable",
        "constant",
        "predicate",
        "conjunction",
        "metadataVal",
        "metadataList",
        "lrnnRule",
        "predicateOffset",
        "predicateMetadata",
        "weightMetadata",
        "templateMetadata",
        "weight",
        "fixedValue",
        "offset",
        "value",
        "number",
        "vector",
        "matrix",
        "dimensions",
        "negation",
    ]

    EOF = Token.EOF
    T__0 = 1
    VARIABLE = 2
    INT = 3
    FLOAT = 4
    ATOMIC_NAME = 5
    IMPLIED_BY = 6
    ASSIGN = 7
    LCURL = 8
    RCURL = 9
    LANGLE = 10
    RANGLE = 11
    LBRACKET = 12
    RBRACKET = 13
    LPAREN = 14
    RPAREN = 15
    COMMA = 16
    SLASH = 17
    CARET = 18
    TRUE = 19
    DOLLAR = 20
    NEGATION = 21
    SPECIAL = 22
    PRIVATE = 23
    WS = 24
    COMMENT = 25
    MULTILINE_COMMENT = 26

    def __init__(self, input: TokenStream, output: TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.8")
        self._interp = ParserATNSimulator(
            self, self.atn, self.decisionsToDFA, self.sharedContextCache
        )
        self._predicates = None

    class TemplateFileContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def templateLine(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.TemplateLineContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.TemplateLineContext, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_templateFile

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTemplateFile"):
                listener.enterTemplateFile(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTemplateFile"):
                listener.exitTemplateFile(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTemplateFile"):
                return visitor.visitTemplateFile(self)
            else:
                return visitor.visitChildren(self)

    def templateFile(self):

        localctx = NeuralogicParser.TemplateFileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_templateFile)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while ((_la) & ~0x3F) == 0 and (
                (1 << _la)
                & (
                    (1 << NeuralogicParser.INT)
                    | (1 << NeuralogicParser.FLOAT)
                    | (1 << NeuralogicParser.ATOMIC_NAME)
                    | (1 << NeuralogicParser.LCURL)
                    | (1 << NeuralogicParser.LANGLE)
                    | (1 << NeuralogicParser.LBRACKET)
                    | (1 << NeuralogicParser.DOLLAR)
                    | (1 << NeuralogicParser.NEGATION)
                    | (1 << NeuralogicParser.SPECIAL)
                    | (1 << NeuralogicParser.PRIVATE)
                )
            ) != 0:
                self.state = 60
                self.templateLine()
                self.state = 65
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TemplateLineContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lrnnRule(self):
            return self.getTypedRuleContext(NeuralogicParser.LrnnRuleContext, 0)

        def fact(self):
            return self.getTypedRuleContext(NeuralogicParser.FactContext, 0)

        def conjunction(self):
            return self.getTypedRuleContext(NeuralogicParser.ConjunctionContext, 0)

        def predicateMetadata(self):
            return self.getTypedRuleContext(
                NeuralogicParser.PredicateMetadataContext, 0
            )

        def predicateOffset(self):
            return self.getTypedRuleContext(NeuralogicParser.PredicateOffsetContext, 0)

        def weightMetadata(self):
            return self.getTypedRuleContext(NeuralogicParser.WeightMetadataContext, 0)

        def templateMetadata(self):
            return self.getTypedRuleContext(NeuralogicParser.TemplateMetadataContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_templateLine

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTemplateLine"):
                listener.enterTemplateLine(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTemplateLine"):
                listener.exitTemplateLine(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTemplateLine"):
                return visitor.visitTemplateLine(self)
            else:
                return visitor.visitChildren(self)

    def templateLine(self):

        localctx = NeuralogicParser.TemplateLineContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_templateLine)
        try:
            self.state = 75
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 1, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 66
                self.lrnnRule()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 67
                self.fact()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 68
                self.conjunction()
                self.state = 69
                self.match(NeuralogicParser.T__0)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 71
                self.predicateMetadata()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 72
                self.predicateOffset()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 73
                self.weightMetadata()
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 74
                self.templateMetadata()
                pass

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ExamplesFileContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def label(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.LabelContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.LabelContext, i)

        def IMPLIED_BY(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.IMPLIED_BY)
            else:
                return self.getToken(NeuralogicParser.IMPLIED_BY, i)

        def liftedExample(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.LiftedExampleContext)
            else:
                return self.getTypedRuleContext(
                    NeuralogicParser.LiftedExampleContext, i
                )

        def getRuleIndex(self):
            return NeuralogicParser.RULE_examplesFile

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterExamplesFile"):
                listener.enterExamplesFile(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitExamplesFile"):
                listener.exitExamplesFile(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitExamplesFile"):
                return visitor.visitExamplesFile(self)
            else:
                return visitor.visitChildren(self)

    def examplesFile(self):

        localctx = NeuralogicParser.ExamplesFileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_examplesFile)
        self._la = 0  # Token type
        try:
            self.state = 90
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 4, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 81
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 77
                    self.label()
                    self.state = 78
                    self.match(NeuralogicParser.IMPLIED_BY)
                    self.state = 79
                    self.liftedExample()
                    self.state = 83
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (
                        (
                            ((_la) & ~0x3F) == 0
                            and (
                                (1 << _la)
                                & (
                                    (1 << NeuralogicParser.INT)
                                    | (1 << NeuralogicParser.FLOAT)
                                    | (1 << NeuralogicParser.ATOMIC_NAME)
                                    | (1 << NeuralogicParser.LCURL)
                                    | (1 << NeuralogicParser.LANGLE)
                                    | (1 << NeuralogicParser.LBRACKET)
                                    | (1 << NeuralogicParser.DOLLAR)
                                    | (1 << NeuralogicParser.NEGATION)
                                    | (1 << NeuralogicParser.SPECIAL)
                                    | (1 << NeuralogicParser.PRIVATE)
                                )
                            )
                            != 0
                        )
                    ):
                        break

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 86
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 85
                    self.liftedExample()
                    self.state = 88
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (
                        (
                            ((_la) & ~0x3F) == 0
                            and (
                                (1 << _la)
                                & (
                                    (1 << NeuralogicParser.INT)
                                    | (1 << NeuralogicParser.FLOAT)
                                    | (1 << NeuralogicParser.ATOMIC_NAME)
                                    | (1 << NeuralogicParser.LCURL)
                                    | (1 << NeuralogicParser.LANGLE)
                                    | (1 << NeuralogicParser.LBRACKET)
                                    | (1 << NeuralogicParser.DOLLAR)
                                    | (1 << NeuralogicParser.NEGATION)
                                    | (1 << NeuralogicParser.SPECIAL)
                                    | (1 << NeuralogicParser.PRIVATE)
                                )
                            )
                            != 0
                        )
                    ):
                        break

                pass

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LiftedExampleContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lrnnRule(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.LrnnRuleContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.LrnnRuleContext, i)

        def conjunction(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.ConjunctionContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.ConjunctionContext, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_liftedExample

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterLiftedExample"):
                listener.enterLiftedExample(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitLiftedExample"):
                listener.exitLiftedExample(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitLiftedExample"):
                return visitor.visitLiftedExample(self)
            else:
                return visitor.visitChildren(self)

    def liftedExample(self):

        localctx = NeuralogicParser.LiftedExampleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_liftedExample)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 94
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 94
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 5, self._ctx)
                if la_ == 1:
                    self.state = 92
                    self.lrnnRule()
                    pass

                elif la_ == 2:
                    self.state = 93
                    self.conjunction()
                    pass

                self.state = 96
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (
                    (
                        ((_la) & ~0x3F) == 0
                        and (
                            (1 << _la)
                            & (
                                (1 << NeuralogicParser.INT)
                                | (1 << NeuralogicParser.FLOAT)
                                | (1 << NeuralogicParser.ATOMIC_NAME)
                                | (1 << NeuralogicParser.LCURL)
                                | (1 << NeuralogicParser.LANGLE)
                                | (1 << NeuralogicParser.LBRACKET)
                                | (1 << NeuralogicParser.DOLLAR)
                                | (1 << NeuralogicParser.NEGATION)
                                | (1 << NeuralogicParser.SPECIAL)
                                | (1 << NeuralogicParser.PRIVATE)
                            )
                        )
                        != 0
                    )
                ):
                    break

            self.state = 98
            self.match(NeuralogicParser.T__0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LabelContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def conjunction(self):
            return self.getTypedRuleContext(NeuralogicParser.ConjunctionContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_label

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterLabel"):
                listener.enterLabel(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitLabel"):
                listener.exitLabel(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitLabel"):
                return visitor.visitLabel(self)
            else:
                return visitor.visitChildren(self)

    def label(self):

        localctx = NeuralogicParser.LabelContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_label)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 100
            self.conjunction()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QueriesFileContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def atom(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.AtomContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.AtomContext, i)

        def IMPLIED_BY(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.IMPLIED_BY)
            else:
                return self.getToken(NeuralogicParser.IMPLIED_BY, i)

        def conjunction(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.ConjunctionContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.ConjunctionContext, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_queriesFile

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterQueriesFile"):
                listener.enterQueriesFile(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitQueriesFile"):
                listener.exitQueriesFile(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitQueriesFile"):
                return visitor.visitQueriesFile(self)
            else:
                return visitor.visitChildren(self)

    def queriesFile(self):

        localctx = NeuralogicParser.QueriesFileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_queriesFile)
        self._la = 0  # Token type
        try:
            self.state = 118
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 9, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 107
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 102
                    self.atom()
                    self.state = 103
                    self.match(NeuralogicParser.IMPLIED_BY)
                    self.state = 104
                    self.conjunction()
                    self.state = 105
                    self.match(NeuralogicParser.T__0)
                    self.state = 109
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (
                        (
                            ((_la) & ~0x3F) == 0
                            and (
                                (1 << _la)
                                & (
                                    (1 << NeuralogicParser.INT)
                                    | (1 << NeuralogicParser.FLOAT)
                                    | (1 << NeuralogicParser.ATOMIC_NAME)
                                    | (1 << NeuralogicParser.LCURL)
                                    | (1 << NeuralogicParser.LANGLE)
                                    | (1 << NeuralogicParser.LBRACKET)
                                    | (1 << NeuralogicParser.DOLLAR)
                                    | (1 << NeuralogicParser.NEGATION)
                                    | (1 << NeuralogicParser.SPECIAL)
                                    | (1 << NeuralogicParser.PRIVATE)
                                )
                            )
                            != 0
                        )
                    ):
                        break

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 114
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 111
                    self.conjunction()
                    self.state = 112
                    self.match(NeuralogicParser.T__0)
                    self.state = 116
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (
                        (
                            ((_la) & ~0x3F) == 0
                            and (
                                (1 << _la)
                                & (
                                    (1 << NeuralogicParser.INT)
                                    | (1 << NeuralogicParser.FLOAT)
                                    | (1 << NeuralogicParser.ATOMIC_NAME)
                                    | (1 << NeuralogicParser.LCURL)
                                    | (1 << NeuralogicParser.LANGLE)
                                    | (1 << NeuralogicParser.LBRACKET)
                                    | (1 << NeuralogicParser.DOLLAR)
                                    | (1 << NeuralogicParser.NEGATION)
                                    | (1 << NeuralogicParser.SPECIAL)
                                    | (1 << NeuralogicParser.PRIVATE)
                                )
                            )
                            != 0
                        )
                    ):
                        break

                pass

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FactContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def atom(self):
            return self.getTypedRuleContext(NeuralogicParser.AtomContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_fact

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFact"):
                listener.enterFact(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFact"):
                listener.exitFact(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFact"):
                return visitor.visitFact(self)
            else:
                return visitor.visitChildren(self)

    def fact(self):

        localctx = NeuralogicParser.FactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_fact)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 120
            self.atom()
            self.state = 121
            self.match(NeuralogicParser.T__0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AtomContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def predicate(self):
            return self.getTypedRuleContext(NeuralogicParser.PredicateContext, 0)

        def weight(self):
            return self.getTypedRuleContext(NeuralogicParser.WeightContext, 0)

        def negation(self):
            return self.getTypedRuleContext(NeuralogicParser.NegationContext, 0)

        def termList(self):
            return self.getTypedRuleContext(NeuralogicParser.TermListContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_atom

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterAtom"):
                listener.enterAtom(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitAtom"):
                listener.exitAtom(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitAtom"):
                return visitor.visitAtom(self)
            else:
                return visitor.visitChildren(self)

    def atom(self):

        localctx = NeuralogicParser.AtomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_atom)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 124
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((_la) & ~0x3F) == 0 and (
                (1 << _la)
                & (
                    (1 << NeuralogicParser.INT)
                    | (1 << NeuralogicParser.FLOAT)
                    | (1 << NeuralogicParser.LCURL)
                    | (1 << NeuralogicParser.LANGLE)
                    | (1 << NeuralogicParser.LBRACKET)
                    | (1 << NeuralogicParser.DOLLAR)
                )
            ) != 0:
                self.state = 123
                self.weight()

            self.state = 127
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.NEGATION:
                self.state = 126
                self.negation()

            self.state = 129
            self.predicate()
            self.state = 131
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.LPAREN:
                self.state = 130
                self.termList()

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TermListContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LPAREN(self):
            return self.getToken(NeuralogicParser.LPAREN, 0)

        def RPAREN(self):
            return self.getToken(NeuralogicParser.RPAREN, 0)

        def term(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.TermContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.TermContext, i)

        def COMMA(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.COMMA)
            else:
                return self.getToken(NeuralogicParser.COMMA, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_termList

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTermList"):
                listener.enterTermList(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTermList"):
                listener.exitTermList(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTermList"):
                return visitor.visitTermList(self)
            else:
                return visitor.visitChildren(self)

    def termList(self):

        localctx = NeuralogicParser.TermListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_termList)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 133
            self.match(NeuralogicParser.LPAREN)
            self.state = 142
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((_la) & ~0x3F) == 0 and (
                (1 << _la)
                & (
                    (1 << NeuralogicParser.VARIABLE)
                    | (1 << NeuralogicParser.INT)
                    | (1 << NeuralogicParser.FLOAT)
                    | (1 << NeuralogicParser.ATOMIC_NAME)
                )
            ) != 0:
                self.state = 134
                self.term()
                self.state = 139
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == NeuralogicParser.COMMA:
                    self.state = 135
                    self.match(NeuralogicParser.COMMA)
                    self.state = 136
                    self.term()
                    self.state = 141
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

            self.state = 144
            self.match(NeuralogicParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TermContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constant(self):
            return self.getTypedRuleContext(NeuralogicParser.ConstantContext, 0)

        def variable(self):
            return self.getTypedRuleContext(NeuralogicParser.VariableContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_term

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTerm"):
                listener.enterTerm(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTerm"):
                listener.exitTerm(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTerm"):
                return visitor.visitTerm(self)
            else:
                return visitor.visitChildren(self)

    def term(self):

        localctx = NeuralogicParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_term)
        try:
            self.state = 148
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [
                NeuralogicParser.INT,
                NeuralogicParser.FLOAT,
                NeuralogicParser.ATOMIC_NAME,
            ]:
                self.enterOuterAlt(localctx, 1)
                self.state = 146
                self.constant()
                pass
            elif token in [NeuralogicParser.VARIABLE]:
                self.enterOuterAlt(localctx, 2)
                self.state = 147
                self.variable()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VariableContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def VARIABLE(self):
            return self.getToken(NeuralogicParser.VARIABLE, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_variable

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterVariable"):
                listener.enterVariable(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitVariable"):
                listener.exitVariable(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitVariable"):
                return visitor.visitVariable(self)
            else:
                return visitor.visitChildren(self)

    def variable(self):

        localctx = NeuralogicParser.VariableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_variable)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(NeuralogicParser.VARIABLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ConstantContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ATOMIC_NAME(self):
            return self.getToken(NeuralogicParser.ATOMIC_NAME, 0)

        def INT(self):
            return self.getToken(NeuralogicParser.INT, 0)

        def FLOAT(self):
            return self.getToken(NeuralogicParser.FLOAT, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_constant

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterConstant"):
                listener.enterConstant(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitConstant"):
                listener.exitConstant(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitConstant"):
                return visitor.visitConstant(self)
            else:
                return visitor.visitChildren(self)

    def constant(self):

        localctx = NeuralogicParser.ConstantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_constant)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 152
            _la = self._input.LA(1)
            if not (
                (
                    ((_la) & ~0x3F) == 0
                    and (
                        (1 << _la)
                        & (
                            (1 << NeuralogicParser.INT)
                            | (1 << NeuralogicParser.FLOAT)
                            | (1 << NeuralogicParser.ATOMIC_NAME)
                        )
                    )
                    != 0
                )
            ):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PredicateContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ATOMIC_NAME(self):
            return self.getToken(NeuralogicParser.ATOMIC_NAME, 0)

        def PRIVATE(self):
            return self.getToken(NeuralogicParser.PRIVATE, 0)

        def SPECIAL(self):
            return self.getToken(NeuralogicParser.SPECIAL, 0)

        def SLASH(self):
            return self.getToken(NeuralogicParser.SLASH, 0)

        def INT(self):
            return self.getToken(NeuralogicParser.INT, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_predicate

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterPredicate"):
                listener.enterPredicate(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitPredicate"):
                listener.exitPredicate(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitPredicate"):
                return visitor.visitPredicate(self)
            else:
                return visitor.visitChildren(self)

    def predicate(self):

        localctx = NeuralogicParser.PredicateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_predicate)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 155
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.PRIVATE:
                self.state = 154
                self.match(NeuralogicParser.PRIVATE)

            self.state = 158
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.SPECIAL:
                self.state = 157
                self.match(NeuralogicParser.SPECIAL)

            self.state = 160
            self.match(NeuralogicParser.ATOMIC_NAME)
            self.state = 163
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.SLASH:
                self.state = 161
                self.match(NeuralogicParser.SLASH)
                self.state = 162
                self.match(NeuralogicParser.INT)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ConjunctionContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def atom(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.AtomContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.AtomContext, i)

        def COMMA(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.COMMA)
            else:
                return self.getToken(NeuralogicParser.COMMA, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_conjunction

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterConjunction"):
                listener.enterConjunction(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitConjunction"):
                listener.exitConjunction(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitConjunction"):
                return visitor.visitConjunction(self)
            else:
                return visitor.visitChildren(self)

    def conjunction(self):

        localctx = NeuralogicParser.ConjunctionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_conjunction)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            self.atom()
            self.state = 170
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 19, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 166
                    self.match(NeuralogicParser.COMMA)
                    self.state = 167
                    self.atom()
                self.state = 172
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 19, self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetadataValContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ATOMIC_NAME(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.ATOMIC_NAME)
            else:
                return self.getToken(NeuralogicParser.ATOMIC_NAME, i)

        def ASSIGN(self):
            return self.getToken(NeuralogicParser.ASSIGN, 0)

        def value(self):
            return self.getTypedRuleContext(NeuralogicParser.ValueContext, 0)

        def DOLLAR(self):
            return self.getToken(NeuralogicParser.DOLLAR, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_metadataVal

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetadataVal"):
                listener.enterMetadataVal(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetadataVal"):
                listener.exitMetadataVal(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetadataVal"):
                return visitor.visitMetadataVal(self)
            else:
                return visitor.visitChildren(self)

    def metadataVal(self):

        localctx = NeuralogicParser.MetadataValContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_metadataVal)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 173
            self.match(NeuralogicParser.ATOMIC_NAME)
            self.state = 174
            self.match(NeuralogicParser.ASSIGN)
            self.state = 180
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [
                NeuralogicParser.INT,
                NeuralogicParser.FLOAT,
                NeuralogicParser.LCURL,
                NeuralogicParser.LBRACKET,
            ]:
                self.state = 175
                self.value()
                pass
            elif token in [NeuralogicParser.ATOMIC_NAME, NeuralogicParser.DOLLAR]:
                self.state = 177
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == NeuralogicParser.DOLLAR:
                    self.state = 176
                    self.match(NeuralogicParser.DOLLAR)

                self.state = 179
                self.match(NeuralogicParser.ATOMIC_NAME)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetadataListContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(NeuralogicParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(NeuralogicParser.RBRACKET, 0)

        def metadataVal(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.MetadataValContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.MetadataValContext, i)

        def COMMA(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.COMMA)
            else:
                return self.getToken(NeuralogicParser.COMMA, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_metadataList

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetadataList"):
                listener.enterMetadataList(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetadataList"):
                listener.exitMetadataList(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetadataList"):
                return visitor.visitMetadataList(self)
            else:
                return visitor.visitChildren(self)

    def metadataList(self):

        localctx = NeuralogicParser.MetadataListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_metadataList)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 182
            self.match(NeuralogicParser.LBRACKET)
            self.state = 191
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.ATOMIC_NAME:
                self.state = 183
                self.metadataVal()
                self.state = 188
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == NeuralogicParser.COMMA:
                    self.state = 184
                    self.match(NeuralogicParser.COMMA)
                    self.state = 185
                    self.metadataVal()
                    self.state = 190
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

            self.state = 193
            self.match(NeuralogicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LrnnRuleContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def atom(self):
            return self.getTypedRuleContext(NeuralogicParser.AtomContext, 0)

        def IMPLIED_BY(self):
            return self.getToken(NeuralogicParser.IMPLIED_BY, 0)

        def conjunction(self):
            return self.getTypedRuleContext(NeuralogicParser.ConjunctionContext, 0)

        def COMMA(self):
            return self.getToken(NeuralogicParser.COMMA, 0)

        def offset(self):
            return self.getTypedRuleContext(NeuralogicParser.OffsetContext, 0)

        def metadataList(self):
            return self.getTypedRuleContext(NeuralogicParser.MetadataListContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_lrnnRule

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterLrnnRule"):
                listener.enterLrnnRule(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitLrnnRule"):
                listener.exitLrnnRule(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitLrnnRule"):
                return visitor.visitLrnnRule(self)
            else:
                return visitor.visitChildren(self)

    def lrnnRule(self):

        localctx = NeuralogicParser.LrnnRuleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_lrnnRule)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 195
            self.atom()
            self.state = 196
            self.match(NeuralogicParser.IMPLIED_BY)
            self.state = 197
            self.conjunction()
            self.state = 200
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.COMMA:
                self.state = 198
                self.match(NeuralogicParser.COMMA)
                self.state = 199
                self.offset()

            self.state = 202
            self.match(NeuralogicParser.T__0)
            self.state = 204
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 25, self._ctx)
            if la_ == 1:
                self.state = 203
                self.metadataList()

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PredicateOffsetContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def predicate(self):
            return self.getTypedRuleContext(NeuralogicParser.PredicateContext, 0)

        def weight(self):
            return self.getTypedRuleContext(NeuralogicParser.WeightContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_predicateOffset

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterPredicateOffset"):
                listener.enterPredicateOffset(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitPredicateOffset"):
                listener.exitPredicateOffset(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitPredicateOffset"):
                return visitor.visitPredicateOffset(self)
            else:
                return visitor.visitChildren(self)

    def predicateOffset(self):

        localctx = NeuralogicParser.PredicateOffsetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_predicateOffset)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 206
            self.predicate()
            self.state = 207
            self.weight()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PredicateMetadataContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def predicate(self):
            return self.getTypedRuleContext(NeuralogicParser.PredicateContext, 0)

        def metadataList(self):
            return self.getTypedRuleContext(NeuralogicParser.MetadataListContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_predicateMetadata

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterPredicateMetadata"):
                listener.enterPredicateMetadata(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitPredicateMetadata"):
                listener.exitPredicateMetadata(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitPredicateMetadata"):
                return visitor.visitPredicateMetadata(self)
            else:
                return visitor.visitChildren(self)

    def predicateMetadata(self):

        localctx = NeuralogicParser.PredicateMetadataContext(
            self, self._ctx, self.state
        )
        self.enterRule(localctx, 36, self.RULE_predicateMetadata)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 209
            self.predicate()
            self.state = 210
            self.metadataList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WeightMetadataContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DOLLAR(self):
            return self.getToken(NeuralogicParser.DOLLAR, 0)

        def ATOMIC_NAME(self):
            return self.getToken(NeuralogicParser.ATOMIC_NAME, 0)

        def metadataList(self):
            return self.getTypedRuleContext(NeuralogicParser.MetadataListContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_weightMetadata

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterWeightMetadata"):
                listener.enterWeightMetadata(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitWeightMetadata"):
                listener.exitWeightMetadata(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitWeightMetadata"):
                return visitor.visitWeightMetadata(self)
            else:
                return visitor.visitChildren(self)

    def weightMetadata(self):

        localctx = NeuralogicParser.WeightMetadataContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_weightMetadata)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 212
            self.match(NeuralogicParser.DOLLAR)
            self.state = 213
            self.match(NeuralogicParser.ATOMIC_NAME)
            self.state = 214
            self.metadataList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TemplateMetadataContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def metadataList(self):
            return self.getTypedRuleContext(NeuralogicParser.MetadataListContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_templateMetadata

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTemplateMetadata"):
                listener.enterTemplateMetadata(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTemplateMetadata"):
                listener.exitTemplateMetadata(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTemplateMetadata"):
                return visitor.visitTemplateMetadata(self)
            else:
                return visitor.visitChildren(self)

    def templateMetadata(self):

        localctx = NeuralogicParser.TemplateMetadataContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_templateMetadata)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 216
            self.metadataList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WeightContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fixedValue(self):
            return self.getTypedRuleContext(NeuralogicParser.FixedValueContext, 0)

        def value(self):
            return self.getTypedRuleContext(NeuralogicParser.ValueContext, 0)

        def DOLLAR(self):
            return self.getToken(NeuralogicParser.DOLLAR, 0)

        def ATOMIC_NAME(self):
            return self.getToken(NeuralogicParser.ATOMIC_NAME, 0)

        def ASSIGN(self):
            return self.getToken(NeuralogicParser.ASSIGN, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_weight

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterWeight"):
                listener.enterWeight(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitWeight"):
                listener.exitWeight(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitWeight"):
                return visitor.visitWeight(self)
            else:
                return visitor.visitChildren(self)

    def weight(self):

        localctx = NeuralogicParser.WeightContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_weight)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 221
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == NeuralogicParser.DOLLAR:
                self.state = 218
                self.match(NeuralogicParser.DOLLAR)
                self.state = 219
                self.match(NeuralogicParser.ATOMIC_NAME)
                self.state = 220
                self.match(NeuralogicParser.ASSIGN)

            self.state = 225
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [NeuralogicParser.LANGLE]:
                self.state = 223
                self.fixedValue()
                pass
            elif token in [
                NeuralogicParser.INT,
                NeuralogicParser.FLOAT,
                NeuralogicParser.LCURL,
                NeuralogicParser.LBRACKET,
            ]:
                self.state = 224
                self.value()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FixedValueContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LANGLE(self):
            return self.getToken(NeuralogicParser.LANGLE, 0)

        def value(self):
            return self.getTypedRuleContext(NeuralogicParser.ValueContext, 0)

        def RANGLE(self):
            return self.getToken(NeuralogicParser.RANGLE, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_fixedValue

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFixedValue"):
                listener.enterFixedValue(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFixedValue"):
                listener.exitFixedValue(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFixedValue"):
                return visitor.visitFixedValue(self)
            else:
                return visitor.visitChildren(self)

    def fixedValue(self):

        localctx = NeuralogicParser.FixedValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_fixedValue)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 227
            self.match(NeuralogicParser.LANGLE)
            self.state = 228
            self.value()
            self.state = 229
            self.match(NeuralogicParser.RANGLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OffsetContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def weight(self):
            return self.getTypedRuleContext(NeuralogicParser.WeightContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_offset

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterOffset"):
                listener.enterOffset(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitOffset"):
                listener.exitOffset(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitOffset"):
                return visitor.visitOffset(self)
            else:
                return visitor.visitChildren(self)

    def offset(self):

        localctx = NeuralogicParser.OffsetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_offset)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 231
            self.weight()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ValueContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def number(self):
            return self.getTypedRuleContext(NeuralogicParser.NumberContext, 0)

        def vector(self):
            return self.getTypedRuleContext(NeuralogicParser.VectorContext, 0)

        def matrix(self):
            return self.getTypedRuleContext(NeuralogicParser.MatrixContext, 0)

        def dimensions(self):
            return self.getTypedRuleContext(NeuralogicParser.DimensionsContext, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_value

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterValue"):
                listener.enterValue(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitValue"):
                listener.exitValue(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitValue"):
                return visitor.visitValue(self)
            else:
                return visitor.visitChildren(self)

    def value(self):

        localctx = NeuralogicParser.ValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_value)
        try:
            self.state = 237
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 28, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 233
                self.number()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 234
                self.vector()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 235
                self.matrix()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 236
                self.dimensions()
                pass

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NumberContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(NeuralogicParser.INT, 0)

        def FLOAT(self):
            return self.getToken(NeuralogicParser.FLOAT, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_number

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterNumber"):
                listener.enterNumber(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitNumber"):
                listener.exitNumber(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitNumber"):
                return visitor.visitNumber(self)
            else:
                return visitor.visitChildren(self)

    def number(self):

        localctx = NeuralogicParser.NumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_number)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 239
            _la = self._input.LA(1)
            if not (_la == NeuralogicParser.INT or _la == NeuralogicParser.FLOAT):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VectorContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(NeuralogicParser.LBRACKET, 0)

        def number(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.NumberContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.NumberContext, i)

        def RBRACKET(self):
            return self.getToken(NeuralogicParser.RBRACKET, 0)

        def COMMA(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.COMMA)
            else:
                return self.getToken(NeuralogicParser.COMMA, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_vector

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterVector"):
                listener.enterVector(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitVector"):
                listener.exitVector(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitVector"):
                return visitor.visitVector(self)
            else:
                return visitor.visitChildren(self)

    def vector(self):

        localctx = NeuralogicParser.VectorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_vector)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 241
            self.match(NeuralogicParser.LBRACKET)
            self.state = 242
            self.number()
            self.state = 247
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == NeuralogicParser.COMMA:
                self.state = 243
                self.match(NeuralogicParser.COMMA)
                self.state = 244
                self.number()
                self.state = 249
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 250
            self.match(NeuralogicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MatrixContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(NeuralogicParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(NeuralogicParser.RBRACKET, 0)

        def vector(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.VectorContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.VectorContext, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_matrix

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMatrix"):
                listener.enterMatrix(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMatrix"):
                listener.exitMatrix(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMatrix"):
                return visitor.visitMatrix(self)
            else:
                return visitor.visitChildren(self)

    def matrix(self):

        localctx = NeuralogicParser.MatrixContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_matrix)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 252
            self.match(NeuralogicParser.LBRACKET)
            self.state = 254
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 253
                self.vector()
                self.state = 256
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la == NeuralogicParser.LBRACKET):
                    break

            self.state = 258
            self.match(NeuralogicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DimensionsContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LCURL(self):
            return self.getToken(NeuralogicParser.LCURL, 0)

        def number(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(NeuralogicParser.NumberContext)
            else:
                return self.getTypedRuleContext(NeuralogicParser.NumberContext, i)

        def RCURL(self):
            return self.getToken(NeuralogicParser.RCURL, 0)

        def COMMA(self, i: int = None):
            if i is None:
                return self.getTokens(NeuralogicParser.COMMA)
            else:
                return self.getToken(NeuralogicParser.COMMA, i)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_dimensions

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterDimensions"):
                listener.enterDimensions(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitDimensions"):
                listener.exitDimensions(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitDimensions"):
                return visitor.visitDimensions(self)
            else:
                return visitor.visitChildren(self)

    def dimensions(self):

        localctx = NeuralogicParser.DimensionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_dimensions)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 260
            self.match(NeuralogicParser.LCURL)
            self.state = 261
            self.number()
            self.state = 266
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == NeuralogicParser.COMMA:
                self.state = 262
                self.match(NeuralogicParser.COMMA)
                self.state = 263
                self.number()
                self.state = 268
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 269
            self.match(NeuralogicParser.RCURL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NegationContext(ParserRuleContext):
        def __init__(
            self, parser, parent: ParserRuleContext = None, invokingState: int = -1
        ):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NEGATION(self):
            return self.getToken(NeuralogicParser.NEGATION, 0)

        def getRuleIndex(self):
            return NeuralogicParser.RULE_negation

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterNegation"):
                listener.enterNegation(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitNegation"):
                listener.exitNegation(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitNegation"):
                return visitor.visitNegation(self)
            else:
                return visitor.visitChildren(self)

    def negation(self):

        localctx = NeuralogicParser.NegationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_negation)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 271
            self.match(NeuralogicParser.NEGATION)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx
