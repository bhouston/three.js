// Exposes ESLint's own core rule implementations to oxlint as a JS plugin,
// so mdcs code style is enforced (and fixed) exactly as ESLint does it.

import { builtinRules } from 'eslint/use-at-your-own-risk';

export default {
	meta: { name: 'eslint-core' },
	rules: Object.fromEntries( builtinRules )
};
