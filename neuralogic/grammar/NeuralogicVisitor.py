# Generated from /home/lukas/Workspace/prcvut/pyneuralogic/neuralogic/grammar/Neuralogic.g4 by ANTLR 4.8
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .NeuralogicParser import NeuralogicParser
else:
    from NeuralogicParser import NeuralogicParser

# This class defines a complete generic visitor for a parse tree produced by NeuralogicParser.


class NeuralogicVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by NeuralogicParser#templateFile.
    def visitTemplateFile(self, ctx: NeuralogicParser.TemplateFileContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#templateLine.
    def visitTemplateLine(self, ctx: NeuralogicParser.TemplateLineContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#examplesFile.
    def visitExamplesFile(self, ctx: NeuralogicParser.ExamplesFileContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#liftedExample.
    def visitLiftedExample(self, ctx: NeuralogicParser.LiftedExampleContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#label.
    def visitLabel(self, ctx: NeuralogicParser.LabelContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#queriesFile.
    def visitQueriesFile(self, ctx: NeuralogicParser.QueriesFileContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#fact.
    def visitFact(self, ctx: NeuralogicParser.FactContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#atom.
    def visitAtom(self, ctx: NeuralogicParser.AtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#termList.
    def visitTermList(self, ctx: NeuralogicParser.TermListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#term.
    def visitTerm(self, ctx: NeuralogicParser.TermContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#variable.
    def visitVariable(self, ctx: NeuralogicParser.VariableContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#constant.
    def visitConstant(self, ctx: NeuralogicParser.ConstantContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#predicate.
    def visitPredicate(self, ctx: NeuralogicParser.PredicateContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#conjunction.
    def visitConjunction(self, ctx: NeuralogicParser.ConjunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#metadataVal.
    def visitMetadataVal(self, ctx: NeuralogicParser.MetadataValContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#metadataList.
    def visitMetadataList(self, ctx: NeuralogicParser.MetadataListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#lrnnRule.
    def visitLrnnRule(self, ctx: NeuralogicParser.LrnnRuleContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#predicateOffset.
    def visitPredicateOffset(self, ctx: NeuralogicParser.PredicateOffsetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#predicateMetadata.
    def visitPredicateMetadata(self, ctx: NeuralogicParser.PredicateMetadataContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#weightMetadata.
    def visitWeightMetadata(self, ctx: NeuralogicParser.WeightMetadataContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#templateMetadata.
    def visitTemplateMetadata(self, ctx: NeuralogicParser.TemplateMetadataContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#weight.
    def visitWeight(self, ctx: NeuralogicParser.WeightContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#fixedValue.
    def visitFixedValue(self, ctx: NeuralogicParser.FixedValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#offset.
    def visitOffset(self, ctx: NeuralogicParser.OffsetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#value.
    def visitValue(self, ctx: NeuralogicParser.ValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#number.
    def visitNumber(self, ctx: NeuralogicParser.NumberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#vector.
    def visitVector(self, ctx: NeuralogicParser.VectorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#matrix.
    def visitMatrix(self, ctx: NeuralogicParser.MatrixContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#dimensions.
    def visitDimensions(self, ctx: NeuralogicParser.DimensionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NeuralogicParser#negation.
    def visitNegation(self, ctx: NeuralogicParser.NegationContext):
        return self.visitChildren(ctx)


del NeuralogicParser
