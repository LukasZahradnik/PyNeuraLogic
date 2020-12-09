# Generated from /home/lukas/Workspace/prcvut/pyneuralogic/neuralogic/grammar/Neuralogic.g4 by ANTLR 4.8
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .NeuralogicParser import NeuralogicParser
else:
    from NeuralogicParser import NeuralogicParser

# This class defines a complete listener for a parse tree produced by NeuralogicParser.
class NeuralogicListener(ParseTreeListener):

    # Enter a parse tree produced by NeuralogicParser#templateFile.
    def enterTemplateFile(self, ctx: NeuralogicParser.TemplateFileContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#templateFile.
    def exitTemplateFile(self, ctx: NeuralogicParser.TemplateFileContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#templateLine.
    def enterTemplateLine(self, ctx: NeuralogicParser.TemplateLineContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#templateLine.
    def exitTemplateLine(self, ctx: NeuralogicParser.TemplateLineContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#examplesFile.
    def enterExamplesFile(self, ctx: NeuralogicParser.ExamplesFileContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#examplesFile.
    def exitExamplesFile(self, ctx: NeuralogicParser.ExamplesFileContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#liftedExample.
    def enterLiftedExample(self, ctx: NeuralogicParser.LiftedExampleContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#liftedExample.
    def exitLiftedExample(self, ctx: NeuralogicParser.LiftedExampleContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#label.
    def enterLabel(self, ctx: NeuralogicParser.LabelContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#label.
    def exitLabel(self, ctx: NeuralogicParser.LabelContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#queriesFile.
    def enterQueriesFile(self, ctx: NeuralogicParser.QueriesFileContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#queriesFile.
    def exitQueriesFile(self, ctx: NeuralogicParser.QueriesFileContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#fact.
    def enterFact(self, ctx: NeuralogicParser.FactContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#fact.
    def exitFact(self, ctx: NeuralogicParser.FactContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#atom.
    def enterAtom(self, ctx: NeuralogicParser.AtomContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#atom.
    def exitAtom(self, ctx: NeuralogicParser.AtomContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#termList.
    def enterTermList(self, ctx: NeuralogicParser.TermListContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#termList.
    def exitTermList(self, ctx: NeuralogicParser.TermListContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#term.
    def enterTerm(self, ctx: NeuralogicParser.TermContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#term.
    def exitTerm(self, ctx: NeuralogicParser.TermContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#variable.
    def enterVariable(self, ctx: NeuralogicParser.VariableContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#variable.
    def exitVariable(self, ctx: NeuralogicParser.VariableContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#constant.
    def enterConstant(self, ctx: NeuralogicParser.ConstantContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#constant.
    def exitConstant(self, ctx: NeuralogicParser.ConstantContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#predicate.
    def enterPredicate(self, ctx: NeuralogicParser.PredicateContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#predicate.
    def exitPredicate(self, ctx: NeuralogicParser.PredicateContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#conjunction.
    def enterConjunction(self, ctx: NeuralogicParser.ConjunctionContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#conjunction.
    def exitConjunction(self, ctx: NeuralogicParser.ConjunctionContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#metadataVal.
    def enterMetadataVal(self, ctx: NeuralogicParser.MetadataValContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#metadataVal.
    def exitMetadataVal(self, ctx: NeuralogicParser.MetadataValContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#metadataList.
    def enterMetadataList(self, ctx: NeuralogicParser.MetadataListContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#metadataList.
    def exitMetadataList(self, ctx: NeuralogicParser.MetadataListContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#lrnnRule.
    def enterLrnnRule(self, ctx: NeuralogicParser.LrnnRuleContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#lrnnRule.
    def exitLrnnRule(self, ctx: NeuralogicParser.LrnnRuleContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#predicateOffset.
    def enterPredicateOffset(self, ctx: NeuralogicParser.PredicateOffsetContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#predicateOffset.
    def exitPredicateOffset(self, ctx: NeuralogicParser.PredicateOffsetContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#predicateMetadata.
    def enterPredicateMetadata(self, ctx: NeuralogicParser.PredicateMetadataContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#predicateMetadata.
    def exitPredicateMetadata(self, ctx: NeuralogicParser.PredicateMetadataContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#weightMetadata.
    def enterWeightMetadata(self, ctx: NeuralogicParser.WeightMetadataContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#weightMetadata.
    def exitWeightMetadata(self, ctx: NeuralogicParser.WeightMetadataContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#templateMetadata.
    def enterTemplateMetadata(self, ctx: NeuralogicParser.TemplateMetadataContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#templateMetadata.
    def exitTemplateMetadata(self, ctx: NeuralogicParser.TemplateMetadataContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#weight.
    def enterWeight(self, ctx: NeuralogicParser.WeightContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#weight.
    def exitWeight(self, ctx: NeuralogicParser.WeightContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#fixedValue.
    def enterFixedValue(self, ctx: NeuralogicParser.FixedValueContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#fixedValue.
    def exitFixedValue(self, ctx: NeuralogicParser.FixedValueContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#offset.
    def enterOffset(self, ctx: NeuralogicParser.OffsetContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#offset.
    def exitOffset(self, ctx: NeuralogicParser.OffsetContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#value.
    def enterValue(self, ctx: NeuralogicParser.ValueContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#value.
    def exitValue(self, ctx: NeuralogicParser.ValueContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#number.
    def enterNumber(self, ctx: NeuralogicParser.NumberContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#number.
    def exitNumber(self, ctx: NeuralogicParser.NumberContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#vector.
    def enterVector(self, ctx: NeuralogicParser.VectorContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#vector.
    def exitVector(self, ctx: NeuralogicParser.VectorContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#matrix.
    def enterMatrix(self, ctx: NeuralogicParser.MatrixContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#matrix.
    def exitMatrix(self, ctx: NeuralogicParser.MatrixContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#dimensions.
    def enterDimensions(self, ctx: NeuralogicParser.DimensionsContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#dimensions.
    def exitDimensions(self, ctx: NeuralogicParser.DimensionsContext):
        pass

    # Enter a parse tree produced by NeuralogicParser#negation.
    def enterNegation(self, ctx: NeuralogicParser.NegationContext):
        pass

    # Exit a parse tree produced by NeuralogicParser#negation.
    def exitNegation(self, ctx: NeuralogicParser.NegationContext):
        pass


del NeuralogicParser
