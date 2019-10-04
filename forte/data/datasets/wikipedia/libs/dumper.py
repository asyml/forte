"""Auxiliary funcitons to dump the data."""
import mako.runtime
import mako.template
from typing import List, Optional


pages_revisions_template = '''
<%!
    from itertools import groupby
    def groupby_action(diff):
        return groupby(diff, lambda d: d.action)
%>
<root>
    % for page in pages:
    <page>
        <title>${page.title}</title>
        <id>${page.id}</id>
        <revisions>
            % for revision in page.revisions:
            <revision>
                <id>${revision.id}</id>
                <user id="{$revision.user.id}" name="${revision.user.text}" />
                <timestamp>${revision.timestamp}</timestamp>
                <references_diff>
                    % for key, group in groupby_action(revision.references_diff):
                    <diff action="${key}">
                        % for _, text in group:
                        <reference>${text}</reference>
                        % endfor
                    </diff>
                    % endfor
                </references_diff>
                <publication_identifiers_diff>
                    % for key, group in groupby_action(revision.publication_identifiers_diff):
                    <diff action="${key}">
                        % for _, identifier in group:
                        <identifier type="${identifier.type}" id="${identifier.id}" />
                        % endfor
                    </diff>
                    % endfor
                </publication_identifiers_diff>
                <sections>
                    % for section in revision.sections:
                    <section level="${section.level}">${section.name}</section>
                    % endfor
                </sections>
                <bibliography>${revision.bibliography}</bibliography>
            </revision>
            %endfor
        </revisions>
    </page>
    % endfor
</root>
'''

stats_template = '''
<stats>
    <performance>
        <start_time>${stats['performance']['start_time']}</start_time>
        <end_time>${stats['performance']['end_time']}</end_time>
        <revisions_analyzed>${stats['performance']['revisions_analyzed']}</revisions_analyzed>
        <pages_analyzed>${stats['performance']['pages_analyzed']}</pages_analyzed>
    </performance>
    <identifiers>
        % for key in ['global', 'last_revision']:
        <${key}>
            % for where, count in stats['identifiers'][key].items():
            <appearance where="${where}" count="${count}" />
            % endfor
        </${key}>
        % endfor
    </identifiers>
</stats>
'''


def render_template(
        template: str,
        output_handler,
        default_filters: Optional[List[str]]=None,
        **kwargs):
    """Render a mako template in the given file."""

    ctx = mako.runtime.Context(output_handler, **kwargs)

    xml_template = mako.template.Template(
        template,
        default_filters=default_filters,
    )
    xml_template.render_context(ctx)


def serialize_page_revisions(pages, output_handler):
    render_template(
        pages_revisions_template,
        output_handler,
        default_fiters=['x'],  # XML escaping
        pages=pages,
    )


def serialize_stats(stats, output_handler):
    render_template(
        stats_template,
        output_handler,
        default_filters=['x'],  # XML escaping
        stats=stats,
    )
