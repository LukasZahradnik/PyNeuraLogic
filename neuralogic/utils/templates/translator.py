from typing import List, Optional, Tuple

from neuralogic.core.constructs.rule import Rule, Metadata
from neuralogic.core.constructs.atom import WeightedAtom
from neuralogic.utils.templates import GCNConv, SAGEConv, GINConv, Embedding, GlobalPooling, TemplateList


def translate_rules(rules):
    rule_context = {}

    for rule in rules:
        head = rule.head
        head_name = str(head.predicate)

        if head_name not in rule_context:
            rule_context[head_name] = {
                "dependent_on": set(),
                "rules": [],
            }

        rule_context[head_name]["rules"].append(rule)

        for atom in rule.body:
            rule_context[head_name]["dependent_on"].add(str(atom.predicate))

    last_layer = None

    for rule_group in rule_context.keys():
        for context in rule_context.values():
            if rule_group in context["dependent_on"]:
                break
        else:
            last_layer = rule_group
            break

    if last_layer is None:
        return None
    return get_template_list(rule_context, last_layer)


def get_template_list(rule_context, last_layer):
    module_list = []

    current_layer = rule_context[last_layer]

    while True:
        rules, metadata = divide_by_type(current_layer["rules"])
        module_info = get_module(rules)

        if module_info is None:
            return None

        module, dependency = module_info

        module_list.append(module)

        if dependency not in rule_context:
            return TemplateList(list(reversed(module_list)))
        current_layer = rule_context[dependency]


def get_module(rules: List[Rule]):
    if len(rules) == 1:
        rule = rules[0]
        for fn in [get_gcn, get_embedding]:
            module_info = fn(rule)
            if module_info is not None:
                return module_info
        return None
    for fn in [get_gin, get_sage, get_pooling]:
        module_info = fn(rules)
        if module_info is not None:
            return module_info
    return None


def divide_by_type(rules):
    metadata = []
    rules_only = []

    for rule in rules:
        if isinstance(rule, Rule):
            rules_only.append(rule)
        elif isinstance(rule, Metadata):
            metadata.append(rule)
    return rules_only, metadata


def get_in_out_from_atom(atom):
    in_channels, out_channels = 1, 1

    if isinstance(atom, WeightedAtom) and isinstance(atom.weight, tuple):
        if len(atom.weight) == 2:
            out_channels, in_channels = atom.weight
        if len(atom.weight) == 1:
            (out_channels,) = atom.weight
    return in_channels, out_channels


def get_embedding(rule: Rule, arity=1) -> Optional[Tuple[Embedding, str]]:
    if len(rule.body) != 1 or len(rule.body[0].terms) != 1 or rule.head.predicate.arity != arity:
        return None

    in_channels, out_channels = get_in_out_from_atom(rule.head)

    module = Embedding(num_embeddings=out_channels, embedding_dim=in_channels, name=rule.head.predicate.name)
    module.features_name = rule.body[0].predicate.name

    return module, str(rule.body[0].predicate)


def get_gcn(rule: Rule) -> Optional[Tuple[GCNConv, str]]:
    if len(rule.body) != 2 or rule.head.predicate.arity != 1:
        return None

    [term] = rule.head.terms
    in_channels, out_channels = get_in_out_from_atom(rule.head)

    if rule.body[0].predicate.arity == 1:
        if rule.body[1].predicate.arity != 2:
            return None
        edge_atom = rule.body[1]
        feature_atom = rule.body[0]
    elif rule.body[0].predicate.arity == 2:
        if rule.body[1].predicate.arity != 1:
            return None
        edge_atom = rule.body[0]
        feature_atom = rule.body[1]
    else:
        return None

    if term not in edge_atom.terms:
        return None
    if feature_atom.terms[0] not in edge_atom.terms:
        return None

    module = GCNConv(in_channels=in_channels, out_channels=out_channels, name=rule.head.predicate.name)
    module.edge_name = edge_atom.predicate.name
    module.features_name = feature_atom.predicate.name

    return module, str(feature_atom.predicate)


def get_sage(rules: List[Rule]) -> Optional[Tuple[SAGEConv, str]]:
    if len(rules) != 2:
        return None

    gcn = None
    embedding = None

    for rule in rules:
        if gcn is None:
            gcn = get_gcn(rule)
        if embedding is None:
            embedding = get_embedding(rule)
        if gcn is not None and embedding is not None:
            break
    else:
        return None

    if embedding[1] != gcn[1]:
        return None

    module = SAGEConv(in_channels=gcn[0].in_channels, out_channels=gcn[0].out_channels, name=embedding[0].name)
    module.edge_name = gcn[0].edge_name
    module.features_name = gcn[0].features_name

    return module, gcn[1]


def get_gin(rules: List[Rule]) -> Optional[Tuple[GINConv, str]]:
    sort_by_name = {}

    for rule in rules:
        name = str(rule.head.predicate)

        if name not in sort_by_name:
            sort_by_name[name] = []
        sort_by_name[name].append(rule)

    sage = None
    embedding = None

    for rule_group in sort_by_name.values():
        if sage is None:
            sage = get_sage(rule_group)
        if embedding is None and len(rule_group) == 1:
            embedding = get_embedding(rule_group[0])
        if sage is not None and embedding is not None:
            break
    else:
        return None

    if embedding[1] != f"{sage[0].name}/1":
        return None

    module = GINConv(
        in_channels=embedding[0].num_embeddings, out_channels=embedding[0].embedding_dim, name=embedding[0].name
    )
    module.edge_name = sage[0].edge_name
    module.features_name = sage[0].features_name

    return module, sage[1]


def get_pooling(rules: List[Rule]) -> Optional[Tuple[GlobalPooling, List[str]]]:
    embeddings = []
    last_embedding = None

    if len(rules) == 0:
        return None

    for rule in rules:
        embedding = get_embedding(rule, 0)
        if embedding is None:
            return None
        embeddings.append(embedding[1])
        last_embedding = embedding[0]

    return (
        GlobalPooling(
            in_channels=last_embedding.embedding_dim,
            out_channels=last_embedding.num_embeddings,
            name=last_embedding.name,
        ),
        embeddings,
    )
