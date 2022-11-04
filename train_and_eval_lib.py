from contrastive import Contrastive
from unsup_embed import UnsupEmbed


def get_trainer(args):
    """Get trainer."""
    if args.method.lower() == 'unsupembed':
        trainer = UnsupEmbed(args)
    elif args.method.lower() == 'contrastive':
        trainer = Contrastive(args)

    return trainer
